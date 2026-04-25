#include "psnd/Model_SHARC_Interface.h"

#include <sys/stat.h>
#include <filesystem>
#include <sys/types.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

#include "psnd/Exception.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/chem.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

const std::string Model_SHARC_Interface::getName() { return "Model_SHARC_Interface"; }

int Model_SHARC_Interface::getType() const { return utils::hash(FUNCTION_NAME); }

#ifdef PSND_WITH_SHARC

namespace py = pybind11;

// Process-wide single interpreter, initialized once. We never call Py_Finalize:
// the OS reclaims it at process exit. This keeps Model_SHARC_Interface safe to
// re-instantiate within a run without risking a double-init crash.
static std::once_flag g_sharc_py_init_flag;
static py::scoped_interpreter* g_sharc_interp = nullptr;

static void sharc_ensure_interpreter(const std::string& sharc_root) {
    std::call_once(g_sharc_py_init_flag, [&]() {
        g_sharc_interp = new py::scoped_interpreter{};
        py::module_ sys = py::module_::import("sys");
        if (!sharc_root.empty()) {
            sys.attr("path").attr("insert")(0, sharc_root + "/bin");
            sys.attr("path").attr("insert")(0, sharc_root + "/lib");
        }
    });
}

static std::string sharc_backend_module_name(SHARCBackend::_type b) {
    switch (b) {
        case SHARCBackend::ORCA:       return "SHARC_ORCA";
        case SHARCBackend::MOLCAS:     return "SHARC_MOLCAS";
        case SHARCBackend::MOLPRO:     return "SHARC_MOLPRO";
        case SHARCBackend::TURBOMOLE:  return "SHARC_TURBOMOLE";
        case SHARCBackend::BAGEL:      return "SHARC_BAGEL";
        case SHARCBackend::GAUSSIAN:   return "SHARC_GAUSSIAN";
        case SHARCBackend::COLUMBUS:   return "SHARC_COLUMBUS";
        case SHARCBackend::NWCHEM:     return "SHARC_NWCHEM";
        case SHARCBackend::PYSCF:      return "SHARC_PYSCF";
        case SHARCBackend::AMS_ADF:    return "SHARC_AMS_ADF";
        case SHARCBackend::MNDO:       return "SHARC_MNDO";
        case SHARCBackend::MOPACPI:    return "SHARC_MOPACPI";
        case SHARCBackend::BASICORCA:  return "SHARC_BASICORCA";
        case SHARCBackend::LVC:        return "SHARC_LVC";
        case SHARCBackend::ANALYTICAL: return "SHARC_ANALYTICAL";
        case SHARCBackend::SPAINN:     return "SHARC_SPAINN";
        case SHARCBackend::SCHNARC:    return "SHARC_SCHNARC";
        case SHARCBackend::QMMM:       return "SHARC_QMMM";
        case SHARCBackend::OPENMM:     return "SHARC_OPENMM";
        case SHARCBackend::DROPLET:    return "SHARC_DROPLET";
        case SHARCBackend::ECI:        return "SHARC_ECI";
        case SHARCBackend::ADAPTIVE:   return "SHARC_ADAPTIVE";
        default:                       return "";
    }
}

// SHARC features whose value type is a list of state indices (parsed from a
// string by _set_driver_requests) — others are treated as plain bool flags.
static bool sharc_feature_is_listlike(const std::string& f) {
    return f == "grad" || f == "nacdr" || f == "overlap" || f == "ion" ||
           f == "multipolar_fit" || f == "density_matrices";
}

// SHARC_FAST subclasses run in-process with persistent=True so model/NN state
// is retained between steps. ab-initio backends run with persistent=False.
static bool sharc_backend_is_fast(SHARCBackend::_type b) {
    switch (b) {
        case SHARCBackend::LVC:
        case SHARCBackend::ANALYTICAL:
        case SHARCBackend::SPAINN:
        case SHARCBackend::SCHNARC:
            return true;
        default:
            return false;
    }
}

#endif  // PSND_WITH_SHARC

void Model_SHARC_Interface::setInputParam_impl(std::shared_ptr<Param> PM) {
#ifndef PSND_WITH_SHARC
    (void)PM;
    throw psnd_error(
        "Model_SHARC_Interface requires -DPSND_ENABLE_SHARC=ON at CMake time. "
        "Rebuild PSiNad with the SHARC option enabled.");
#else
    Kernel_Representation::onthefly = true;

    std::string backend_str = _param->get_string({"model.sharc_backend"}, LOC(), "ANALYTICAL");
    backend                 = SHARCBackend::_dict.at(backend_str);
    backend_name            = sharc_backend_module_name(backend);
    if (backend_name.empty())
        throw psnd_error("Model_SHARC_Interface: unknown sharc_backend '" + backend_str + "'");

    template_file  = _param->get_string({"model.sharc_template"},  LOC(), backend_str + ".template");
    resources_file = _param->get_string({"model.sharc_resources"}, LOC(), backend_str + ".resources");
    qmin_file      = _param->get_string({"model.sharc_qmin"},      LOC(), "QM.in");

    const char* sharc_env = std::getenv("SHARC");
    sharc_root = _param->get_string({"model.sharc_root"}, LOC(), sharc_env ? sharc_env : "");

    // features are derived from QM.in's `## Requests` block at initialize time
    // (see initializeKernel_impl). QM.in is the single source of truth, matching
    // SHARC's own convention where the Fortran driver writes QM.in and the
    // backend reads it — no duplicate declaration in input.json.
    features.clear();
    has_soc    = false;  // recomputed from QMin.requests after read_requests(file)
    has_mm     = (backend == SHARCBackend::QMMM || backend == SHARCBackend::OPENMM ||
                  backend == SHARCBackend::DROPLET || backend == SHARCBackend::ECI);
    persistent = sharc_backend_is_fast(backend);

    // sharc_keep_scratch: when true, QC scratch stays at `-d/scratch` (persistent,
    // easy to inspect inp/out/err). When false (default), use an atomically-unique
    // mkdtemp path `/tmp/psinad_sharc_XXXXXX` and rm -rf at finalize. Uniqueness
    // matters because multiple PSiNad processes run in parallel as an ensemble
    // and a fixed /tmp path would collide.
    keep_scratch = _param->get_bool({"model.sharc_keep_scratch"}, LOC(), false);
#endif
}

void Model_SHARC_Interface::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
#ifndef PSND_WITH_SHARC
    (void)DS;
    throw psnd_error("Model_SHARC_Interface requires -DPSND_ENABLE_SHARC=ON");
#else
    x    = DS->def(DATA::integrator::x);
    p    = DS->def(DATA::integrator::p);
    x0   = DS->def(DATA::model::x0);
    p0   = DS->def(DATA::model::p0);
    atoms = DS->def(DATA::model::atoms);
    mass  = DS->def(DATA::model::mass);
    vpes  = DS->def(DATA::model::vpes);
    grad  = DS->def(DATA::model::grad);
    V     = DS->def(DATA::model::V);
    dV    = DS->def(DATA::model::dV);
    eig   = DS->def(DATA::model::rep::eig);
    T     = DS->def(DATA::model::rep::T);
    dE    = DS->def(DATA::model::rep::dE);
    nac   = DS->def(DATA::model::rep::nac);
    nac_prev = DS->def(DATA::model::rep::nac_prev);

    succ_ptr         = DS->def(DATA::control::succ);
    last_attempt_ptr = DS->def(DATA::control::last_attempt);
    frez_ptr         = DS->def(DATA::control::frez);
    fail_type_ptr    = DS->def(DATA::control::fail_type);
    dt_ptr           = DS->def(DATA::control::dt);
    t_ptr            = DS->def(DATA::control::t);
    istep_ptr        = DS->def(DATA::control::istep);

    ARRAY_EYE(T.data(), Dimension::F);

    // Always bind complex-rep spans. Model fills Vc + dVc (+ real `nac` above);
    // Kernel_Representation derives Ec/Tc/dEc from those. Ec/Tc/dEc are bound
    // here only for record rules and debug introspection.
    Vc  = DS->def(DATA::model::Vc);
    dVc = DS->def(DATA::model::dVc);
    dEc = DS->def(DATA::model::rep::dEc);
    Ec  = DS->def(DATA::model::rep::Ec);
    Tc  = DS->def(DATA::model::rep::Tc);

    // atoms[] and mass[] must be filled HERE — Kernel_Update_x::setInputDataSet
    // reads mass and pre-computes minv = 1/mass at this stage. We only need the
    // element labels (no unit, no coords), so read just the QM.in header.
    // x0/p0 are populated later in initializeKernel_impl, where SHARC's
    // already-parsed `factor` does the unit conversion for us.
    std::ifstream ifs(qmin_file);
    if (!ifs.good()) throw psnd_error("Model_SHARC_Interface: cannot open " + qmin_file);
    std::string line;
    std::getline(ifs, line);
    std::stringstream(line) >> natom;
    psnd_assert(natom * 3 == (int)Dimension::N, "Dimension mismatch: 3*natom != N");
    std::getline(ifs, line);  // comment line
    for (int i = 0, idx = 0; i < natom; ++i) {
        std::string elem;
        ifs >> elem;
        atoms[i] = chem::getElemIndex(elem);
        double dummy;
        for (int a = 0; a < 3; ++a, ++idx) {
            ifs >> dummy;  // skip coord; real value loaded in initializeKernel
            mass[idx] = chem::getElemMass(atoms[i]) / phys::au_2_amu;
        }
    }
#endif
}

Status& Model_SHARC_Interface::initializeKernel_impl(Status& stat) {
#ifndef PSND_WITH_SHARC
    throw psnd_error("Model_SHARC_Interface requires -DPSND_ENABLE_SHARC=ON");
#else
    sharc_ensure_interpreter(sharc_root);
    setup_interface_calls = 0;

    numpy_mod = py::module_::import("numpy");

    py::module_ mod = py::module_::import(backend_name.c_str());
    // SHARC exposes the class under the same symbol as the module name, e.g.
    // SHARC_ANALYTICAL.py defines `class SHARC_ANALYTICAL(SHARC_FAST)`.
    if (!py::hasattr(mod, backend_name.c_str())) {
        throw psnd_error(
            "Model_SHARC_Interface: module " + backend_name +
            " does not expose a class named " + backend_name);
    }
    py::object klass = mod.attr(backend_name.c_str());

    sharc_instance = klass(py::arg("persistent") = persistent);

    // Note: we used to monkey-patch setup_interface with a py::cpp_function to
    // count calls and verify the "exactly once per trajectory" invariant (plan
    // § 3.4). But SHARC_ABINITIO backends dispatch per-state jobs through
    // multiprocessing.Pool, which pickles the SHARC instance; a py::cpp_function
    // member carries a PyCapsule that can't be pickled. Since we invoke
    // setup_interface exactly once below and SHARC never self-calls it, the
    // invariant is enforced by code structure — no runtime check needed.

    // Follow SHARC's standard init order.
    sharc_instance.attr("setup_mol")(qmin_file);
    sharc_instance.attr("read_resources")(resources_file);
    sharc_instance.attr("read_template")(template_file);

    // SHARC has now parsed QM.in's `unit` keyword into molecule["factor"]
    // (input → Bohr). Cache its inverse for per-step coord scaling, then read
    // the geom block from QM.in to fill x0 — coords aren't stored by setup_mol
    // (they normally arrive via set_coords each step).
    py::object qmin_mol = sharc_instance.attr("QMin").attr("molecule");
    double     factor   = py::float_(qmin_mol["factor"]);
    coord_scale         = 1.0 / factor;

    std::ifstream ifs(qmin_file);
    if (!ifs.good())
        throw psnd_error("Model_SHARC_Interface: cannot open " + qmin_file);
    std::string line;
    std::getline(ifs, line);  // line 1: natom (already validated in setInputDataSet)
    std::getline(ifs, line);  // line 2: comment
    for (int i = 0, idx = 0; i < natom; ++i) {
        std::string elem;
        ifs >> elem;
        for (int a = 0; a < 3; ++a, ++idx) {
            double v;
            ifs >> v;
            x0[idx] = v * factor;
            p0[idx] = 0.0;
        }
    }

    // Root SHARC's savedir under PSiNad's working directory (the -d gflag /
    // Kernel::directory). Scratchdir is either persistent there (keep_scratch)
    // or a unique /tmp/psinad_sharc_XXXXXX (discardable, rm at finalize).
    //
    // Scope note: one PSiNad process runs exactly one trajectory. Ensembles are
    // launched as separate processes (each with its own -d).
    //
    // Paths must be absolute: SHARC's multiprocessing workers os.chdir into the
    // workdir, so any relative path is resolved against the worker's CWD (not
    // our parent process) and blows up with FileNotFoundError.
    {
        auto ensure_dir = [](const std::string& path) {
            struct ::stat st {};
            if (::stat(path.c_str(), &st) != 0) {
                if (::mkdir(path.c_str(), 0755) != 0) {
                    throw psnd_error("Model_SHARC_Interface: cannot mkdir " + path);
                }
            }
        };
        ensure_dir(directory);
        std::string abs_directory = std::filesystem::absolute(directory).string();
        if (keep_scratch) {
            scratch_dir = abs_directory + "/scratch";
            ensure_dir(scratch_dir);
        } else {
            // mkdtemp atomically creates a unique directory; safe against
            // races when many ensemble processes start simultaneously.
            char tmpl[] = "/tmp/psinad_sharc_XXXXXX";
            if (::mkdtemp(tmpl) == nullptr) {
                throw psnd_error("Model_SHARC_Interface: mkdtemp failed for /tmp/psinad_sharc_XXXXXX");
            }
            scratch_dir = tmpl;
        }
        std::string sharc_savedir = abs_directory + "/SAVE";
        ensure_dir(sharc_savedir);
        sharc_instance.attr("QMin").attr("resources")["scratchdir"] = scratch_dir;
        sharc_instance.attr("QMin").attr("save")["savedir"] = sharc_savedir;
    }

    sharc_instance.attr("setup_interface")();
    setup_interface_calls = 1;

    // Let SHARC parse QM.in's `## Requests` block (file form of read_requests).
    // QM.in is the single source of truth for which quantities we ask for —
    // matches SHARC's own convention (see sharc_architecture_notes.md § 3.4).
    sharc_instance.attr("read_requests")(qmin_file);

    // Introspect QMin.requests to build our C++ features set. We recognise the
    // per-step output features; control keys (retain, step, init, ...) are
    // skipped. Anything truthy in the dict is kept.
    {
        py::object qmin_req   = sharc_instance.attr("QMin").attr("requests");
        py::object req_data   = qmin_req.attr("data");
        static const std::vector<std::string> known_features = {
            "h", "dm", "soc", "grad", "nacdr", "phases",
            "dmdr", "socdr", "overlap", "ion",
            "multipolar_fit", "density_matrices",
        };
        features.clear();
        for (const auto& f : known_features) {
            if (!req_data.contains(py::str(f))) continue;
            py::object val = req_data[py::str(f)];
            if (!val.is_none() && py::bool_(val)) features.insert(f);
        }
        has_soc = features.count("soc") > 0;
    }

    // Pull nstates from the Python side for later QMout shape checks.
    py::object mol = sharc_instance.attr("QMin").attr("molecule");
    nstates        = py::int_(mol["nmstates"]);
    if (nstates != (int)Dimension::F)
        throw psnd_error("Model_SHARC_Interface: SHARC nmstates != PSiNad Dimension::F");

    return stat;
#endif
}

Status& Model_SHARC_Interface::executeKernel_impl(Status& stat) {
#ifndef PSND_WITH_SHARC
    throw psnd_error("Model_SHARC_Interface requires -DPSND_ENABLE_SHARC=ON");
#else
    if (stat.frozen) return stat;

    // PSiNad's x is Bohr; SHARC.set_coords will multiply by molecule["factor"]
    // and store Bohr internally. We pre-divide by factor (cached as coord_scale
    // at init time) so the round-trip lands back at the original Bohr values
    // regardless of QM.in's declared unit.
    py::array_t<double> coords({(py::ssize_t)natom, (py::ssize_t)3});
    auto raw = coords.mutable_unchecked<2>();
    for (int iatom = 0, idx = 0; iatom < natom; ++iatom) {
        for (int a = 0; a < 3; ++a, ++idx) {
            raw(iatom, a) = x[idx] * coord_scale;
        }
    }

    // Bump the SHARC step counter so its save/newstep logic tracks us.
    py::dict step_req;
    step_req["tasks"] = std::string("step ") + std::to_string(istep_ptr[0]);
    for (const auto& f : features) {
        if (sharc_feature_is_listlike(f)) {
            step_req[py::str(f)] = std::string("all");
        } else {
            step_req[py::str(f)] = true;
        }
    }
    sharc_instance.attr("read_requests")(step_req);

    sharc_instance.attr("set_coords")(coords);
    sharc_instance.attr("run")();
    py::object qmout = sharc_instance.attr("getQMout")();

        // --- unpack: real path (no SOC) ---
        // qmout["h"] — (F, F) complex. For no-SOC case imaginary part should be 0.
        // Diagonal → eig, off-diagonal → dE (real part only, since real path).
        py::array_t<std::complex<double>> hmat = py::cast<py::array_t<std::complex<double>>>(qmout["h"]);
        auto h_unchecked = hmat.unchecked<2>();
        for (int i = 0; i < (int)Dimension::F; ++i) {
            eig[i] = h_unchecked(i, i).real();
        }

        // Only fetch qmout entries we actually asked for; QMout returns None
        // for unrequested keys and pybind11 will throw trying to cast that to
        // an array. Zero-fill the corresponding spans so downstream kernels see
        // defined values even when a feature is disabled.
        const bool have_grad  = features.count("grad")  > 0;
        const bool have_nacdr = features.count("nacdr") > 0;

        if (has_soc) {
            // --- General_soc path ---
            // Model fills: Vc (complex H with SOC), dVc (complex d/dR Vc) and
            // the shared real `nac` span (populated from SHARC's `nacdr`).
            // Kernel_Representation then computes Ec/Tc via EigenSolve(Vc) and
            // dEc = dVc + [nac, Vc] — see Kernel_Representation.cpp:328-388.
            // Writing Ec/Tc/dEc here would be wasted work (they get overwritten).
            for (int i = 0; i < (int)Dimension::F; ++i) {
                for (int j = 0; j < (int)Dimension::F; ++j) {
                    Vc[i * Dimension::F + j] = h_unchecked(i, j);
                }
            }

            for (int j = 0; j < (int)Dimension::NFF; ++j) {
                dVc[j] = psnd_complex{0.0, 0.0};
                nac[j] = 0.0;
            }
            if (have_grad || have_nacdr) {
                auto g = have_grad  ? py::cast<py::array_t<double>>(qmout["grad"]).unchecked<3>()
                                    : py::array_t<double>({1, 1, 1}).unchecked<3>();
                auto n = have_nacdr ? py::cast<py::array_t<double>>(qmout["nacdr"]).unchecked<4>()
                                    : py::array_t<double>({1, 1, 1, 1}).unchecked<4>();
                // dVc diagonal = grad (real part). Off-diagonal dVc = d(SOC)/dR
                // which SHARC_LVC does not export (no `socdr`); the NAC part of
                // the derivative flows through `nac` and gets added as [nac, Vc]
                // in Kernel_Representation.
                for (int k = 0, idx3 = 0; k < natom; ++k) {
                    for (int a = 0; a < 3; ++a, ++idx3) {
                        for (int i = 0; i < (int)Dimension::F; ++i) {
                            for (int j = 0; j < (int)Dimension::F; ++j) {
                                int flat = idx3 * Dimension::FF + i * Dimension::F + j;
                                if (i == j && have_grad) {
                                    dVc[flat] = psnd_complex{g(i, k, a), 0.0};
                                } else if (i != j && have_nacdr) {
                                    nac[flat] = n(i, j, k, a);
                                }
                            }
                        }
                    }
                }
            }
            if (!features.count("phases") && stat.first_step && have_nacdr) {
                for (int i = 0; i < (int)Dimension::NFF; ++i) nac_prev[i] = nac[i];
            }
        } else {
            // --- real path ---
            for (int j = 0; j < (int)Dimension::NFF; ++j) { dE[j] = 0.0; nac[j] = 0.0; }
            if (have_grad || have_nacdr) {
                auto g = have_grad  ? py::cast<py::array_t<double>>(qmout["grad"]).unchecked<3>()
                                    : py::array_t<double>({1, 1, 1}).unchecked<3>();
                auto n = have_nacdr ? py::cast<py::array_t<double>>(qmout["nacdr"]).unchecked<4>()
                                    : py::array_t<double>({1, 1, 1, 1}).unchecked<4>();
                // dE[idx3, i, i] = grad_i(idx3);  dE[idx3, i, j] = nac(i,j,idx3)*(eig[j]-eig[i])
                for (int k = 0, idx3 = 0; k < natom; ++k) {
                    for (int a = 0; a < 3; ++a, ++idx3) {
                        for (int i = 0; i < (int)Dimension::F; ++i) {
                            for (int j = 0; j < (int)Dimension::F; ++j) {
                                int flat = idx3 * Dimension::FF + i * Dimension::F + j;
                                if (i == j && have_grad) {
                                    dE[flat] = g(i, k, a);
                                } else if (i != j && have_nacdr) {
                                    nac[flat] = n(i, j, k, a);
                                    dE[flat]  = nac[flat] * (eig[j] - eig[i]);
                                }
                            }
                        }
                    }
                }
            }
            // SHARC can track phases internally via the 'phases' feature; if the
            // user didn't request it, fall back to the same sign-tracking logic
            // Model_QMInterface uses so NAC continuity is preserved across steps.
            // (track_nac_sign extraction to util/nac_phase.h is a § 3.4 TODO.)
            if (!features.count("phases") && stat.first_step && have_nacdr) {
                for (int i = 0; i < (int)Dimension::NFF; ++i) nac_prev[i] = nac[i];
            }
        }

    // Mirror SHARC's standalone-driver post-step sequence. For FAST backends
    // in persistent mode, write_step_file only updates the in-memory
    // `savedict["last_step"]` — without it, the next call's _step_logic sees a
    // stale last_step and raises "Determined last step (0) from savedir and
    // specified step (N) do not fit". Any exception from these steps should
    // propagate — failures here are programming bugs, not recoverable states.
    sharc_instance.attr("clean_savedir")();
    sharc_instance.attr("create_restart_files")();
    sharc_instance.attr("write_step_file")();

    stat.succ = true;
    if (stat.fail_type == 1 && !stat.last_attempt) stat.fail_type = 0;
    return stat;
#endif
}

Status& Model_SHARC_Interface::finalizeKernel_impl(Status& stat) {
#ifdef PSND_WITH_SHARC
    // Single-trajectory scope invariant (docs/dev/sharc_interface_plan.md § 3.4):
    // setup_interface must have been called exactly once. If not, something
    // in the Model lifecycle was re-entered and we need to know.
    if (setup_interface_calls != 1) {
        std::cerr << "[SHARC] WARNING: setup_interface was called "
                  << setup_interface_calls
                  << " times (expected 1 for single-trajectory scope)\n";
    }
    try {
        if (!sharc_instance.is_none()) {
            sharc_instance.attr("create_restart_files")();
        }
    } catch (const py::error_already_set& e) {
        std::cerr << "[SHARC] create_restart_files failed: " << e.what() << std::endl;
    }
    // Do NOT Py_Finalize; the scoped_interpreter is process-wide (see
    // sharc_ensure_interpreter) and will be cleaned up by the OS.
    sharc_instance = py::object();

    // Discard /tmp scratchdir when user didn't opt in to keeping it. Only
    // applies to the mkdtemp path; when keep_scratch is true scratch_dir lives
    // under -d and is preserved for inspection. Best-effort — cleanup errors
    // are logged but not thrown (finalize should be idempotent).
    if (!keep_scratch && !scratch_dir.empty() &&
        scratch_dir.rfind("/tmp/psinad_sharc_", 0) == 0) {
        std::error_code ec;
        std::filesystem::remove_all(scratch_dir, ec);
        if (ec) {
            std::cerr << "[SHARC] scratch cleanup failed at " << scratch_dir
                      << ": " << ec.message() << std::endl;
        }
    }
#endif
    return stat;
}

};  // namespace PROJECT_NS
