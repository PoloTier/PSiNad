#include "psnd/Sampling_Elec.h"

#include "psnd/Kernel_Elec_Utils.h"
#include "psnd/Kernel_NAForce.h"
#include "psnd/Kernel_Random.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/debug_utils.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

#include <string>
#include <vector>

namespace PROJECT_NS {

inline bool isFileExists(const std::string& name) { return std::ifstream{name.c_str()}.good(); }

inline std::string trim_copy(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

const std::string Sampling_Elec::getName() { return "Sampling_Elec"; }

int Sampling_Elec::getType() const { return utils::hash(FUNCTION_NAME); }

void Sampling_Elec::setInputParam_impl(std::shared_ptr<Param> PM) {
    // for c & rho_ele
    sampling_type = ElectronicSamplingPolicy::_from(
        _param->get_string({"solver.sampling_ele_flag", "solver.sampling.ele_flag"}, LOC(), "Constraint"));

    sampling_file      = _param->get_string({"solver.sampling_file", "solver.sampling.file"}, LOC(), "init");

    occ0 = _param->get_int({"model.occ", "solver.occ"}, LOC(), -1);
    if (occ0 < 0) throw std::runtime_error("occ < 0");
    if (occ0 >= Dimension::F) throw std::runtime_error("occ >= F");

    gamma1 = _param->get_real({"solver.gamma"}, LOC(), elec_utils::gamma_wigner(Dimension::F));
    if (gamma1 < -1.5) gamma1 = elec_utils::gamma_opt(Dimension::F);     // gammaR using gamma=-2
    if (gamma1 < -0.5) gamma1 = elec_utils::gamma_wigner(Dimension::F);  // gammaW using gamma=-1
    xi1 = (1 + Dimension::F * gamma1);

    // for rho_nuc
    use_cv   = _param->get_bool({"solver.use_cv"}, LOC(), false);
    use_wmm  = _param->get_bool({"solver.use_wmm"}, LOC(), false);
    use_sum  = _param->get_bool({"solver.use_sum"}, LOC(), false);
    use_fssh = _param->get_bool({"solver.use_fssh"}, LOC(), false);
}

void Sampling_Elec::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    c       = DS->def(DATA::integrator::c);        // wave function of electrons
    rho_ele = DS->def(DATA::integrator::rho_ele);  // density matrix of electrons
    rho_nuc = DS->def(DATA::integrator::rho_nuc);  // density matrix of electrons but used for nuclear force (EX.
                                                   // F_EHR=-Tr[rho_nuc @ \nabla H])
    occ_nuc = DS->def(DATA::integrator::occ_nuc);  // which electron state is occupied
    w       = DS->def(DATA::integrator::w);
    T       = DS->def(DATA::model::rep::T);
    Tc      = DS->def(DATA::model::rep::Tc);
}

Status& Sampling_Elec::initializeKernel_impl(Status& stat) { return stat; }

Status& Sampling_Elec::executeKernel_impl(Status& stat) {
    // if restart, we should get all initial values and current values!
    // not that: all values defined by Kernel_Elec_Functions will also be recoveed later.
    // not that: all values defined by Kernel_Recorder will also be recovered later.
    const std::string load_str = _param->get_string({"load", "solver.load"}, LOC(), "");
    if (load_str.find(":continue") != std::string::npos) return stat;
    if (load_str.find(":resume") != std::string::npos) return stat;
    if (load_str.find(":restart") != std::string::npos) {  //
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        _dataset->def(DATA::init::c, _dataset_load);
        _dataset->def(DATA::init::rho_ele, _dataset_load);
        _dataset->def(DATA::init::rho_nuc, _dataset_load);
        _dataset->def(DATA::init::T, _dataset_load);
        _dataset->def(DATA::integrator::c, _dataset_load);
        _dataset->def(DATA::integrator::rho_ele, _dataset_load);
        _dataset->def(DATA::integrator::rho_nuc, _dataset_load);
        _dataset->def(DATA::integrator::occ_nuc, _dataset_load);
        _dataset->def(DATA::integrator::w, _dataset_load);
        return stat;
    }

    for (int iP = 0; iP < Dimension::P; ++iP) {  // use P insread P_NOW
        auto c       = this->c.subspan(iP * Dimension::F, Dimension::F);
        auto rho_ele = this->rho_ele.subspan(iP * Dimension::FF, Dimension::FF);
        auto rho_nuc = this->rho_nuc.subspan(iP * Dimension::FF, Dimension::FF);
        auto w       = this->w.subspan(iP, 1);
        auto T       = this->T.subspan(iP * Dimension::FF, Dimension::FF);
        auto occ_nuc = this->occ_nuc.subspan(iP, 1);

        /////////////////////////////////////////////////////////////////

        int iocc;
        Kernel_Random::rand_catalog(&iocc, 1, true, 0, Dimension::F - 1);
        iocc = ((use_sum) ? iocc : occ0);
        w[0] = (use_sum) ? psnd_complex(Dimension::F) : phys::math::iu;

        // Sampling pipeline:
        //   1) sample mapping amplitudes c or directly construct rho_ele
        //   2) build rho_ele in inp_repr:
        //        rho_ele = |c><c|  for the branches calling ker_from_c(...)
        //      so Tr(rho_ele) = sum_i |c_i|^2, which is not always 1 for window-style sampling
        //   3) build rho_nuc from rho_ele for nuclear-force / occupation usage:
        //        rho_nuc = xi * rho_ele - gamma * I
        //      unless use_cv=true, in which case the diagonal is quantized to a one-hot occupation
        //   4) after sampling, occ_nuc is determined from rho_nuc in nuc_repr
        switch (sampling_type) {
            case ElectronicSamplingPolicy::Focus: {
                // Focus sampling gives a pure-state rho_ele = |c><c|.
                // rho_nuc is then the affine-mapped density used by the nuclear part.
                elec_utils::c_focus(c.data(), xi1, gamma1, iocc, Dimension::F);
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), xi1, gamma1, Dimension::F, use_cv, iocc);
                break;
            }
            case ElectronicSamplingPolicy::GDTWA: {
                // In GDTWA, rho_ele is constructed directly instead of from c.
                // Final shape:
                //   - diagonal: occupied state = 1, others = 0
                //   - off-diagonal: phase-sampled coherence around iocc
                // rho_nuc here is copied from rho_ele because xi=1 and gamma=0.
                elec_utils::c_focus(c.data(), xi1, gamma1, iocc, Dimension::F);  // @useless

                /// GDTWA sampling step 1: discrete random phase
                for (int j = 0; j < Dimension::F; ++j) {
                    if (j == iocc) {
                        rho_ele[j * Dimension::Fadd1] = 1.0e0;
                    } else {
                        double randu;
                        Kernel_Random::rand_uniform(&randu);
                        randu                            = phys::math::halfpi * (int(randu / 0.25f) + 0.5);
                        rho_ele[iocc * Dimension::F + j] = cos(randu) + phys::math::im * sin(randu);
                        rho_ele[j * Dimension::F + iocc] = std::conj(rho_ele[iocc * Dimension::F + j]);
                    }
                }
                for (int i = 0, ij = 0; i < Dimension::F; ++i) {
                    for (int j = 0; j < Dimension::F; ++j, ++ij) {
                        if (i == iocc || j == iocc) continue;
                        rho_ele[ij] = rho_ele[iocc * Dimension::F + j] / rho_ele[iocc * Dimension::F + i];
                    }
                }
                /// GDTWA sampling step 2: set off-diagonal
                double gamma_ou = phys::math::sqrthalf;
                double gamma_uu = 0.0e0;
                for (int i = 0, ij = 0; i < Dimension::F; ++i) {
                    for (int j = 0; j < Dimension::F; ++j, ++ij) {
                        if (i == j) {
                            rho_ele[ij] = (i == iocc) ? phys::math::iu : phys::math::iz;
                        } else if (i == iocc || j == iocc) {
                            rho_ele[ij] *= gamma_ou;
                        } else {
                            rho_ele[ij] *= gamma_uu;
                        }
                    }
                }
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), 1, 0, Dimension::F, false, iocc);
                break;
            }
            case ElectronicSamplingPolicy::SQCtri: {
                // Window sampling generally gives a non-unit-norm c.
                // Hence rho_ele = |c><c| is not guaranteed to have trace 1.
                // rho_nuc is the affine-mapped density used later by the nuclear kernels.
                elec_utils::c_window(c.data(), iocc, ElectronicSamplingPolicy::SQCtri, Dimension::F);
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), 1.0, gamma1, Dimension::F, use_cv, iocc);
                break;
            }
            case ElectronicSamplingPolicy::SQCspx: {
                // Same overall data flow as SQCtri:
                //   c -> rho_ele = |c><c| -> rho_nuc = rho_ele - gamma * I (or quantized diagonal if use_cv=true)
                elec_utils::c_sphere(c.data(), Dimension::F);
                for (int i = 0; i < Dimension::F; ++i) c[i] = std::abs(c[i] * c[i]);
                c[iocc] += 1.0e0;
                for (int i = 0; i < Dimension::F; ++i) {
                    psnd_real randu;
                    Kernel_Random::rand_uniform(&randu);
                    randu *= phys::math::twopi;
                    c[i] = sqrt(c[i]) * (cos(randu) + phys::math::im * sin(randu));
                }
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), 1.0, gamma1, Dimension::F, use_cv, iocc);
                break;
            }
            case ElectronicSamplingPolicy::SQCtest01: {
                // xi1 = 1 + F * gamma.
                // For window sampling, Tr(rho_ele) is generally not 1.
                // Here we compensate that by using xi_eff = xi1 / Tr(rho_ele), so that
                // rho_nuc = xi_eff * rho_ele - gamma * I
                // has Tr(rho_nuc) = 1 when use_cv=false.
                elec_utils::c_window(c.data(), iocc, ElectronicSamplingPolicy::SQCtri, Dimension::F);
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                double tr_rho_ele = 0.0;
                for (int i = 0; i < Dimension::F; ++i) tr_rho_ele += std::abs(rho_ele[i * Dimension::Fadd1]);
                double xi_eff = xi1 / tr_rho_ele;
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), xi_eff, gamma1, Dimension::F, use_cv, iocc);
                break;
            }
            case ElectronicSamplingPolicy::SQCtest02: {
                // In this branch c is explicitly normalized first.
                // Therefore rho_ele = |c><c| has Tr(rho_ele) = 1, and rho_nuc is then
                // built with an effective gamma inferred from the sampled norm.
                elec_utils::c_window(c.data(), iocc, ElectronicSamplingPolicy::SQCtri, Dimension::F);
                double norm = 0.0e0;
                for (int i = 0; i < Dimension::F; ++i) norm += std::abs(c[i] * c[i]);
                xi1    = norm;
                gamma1 = (xi1 - 1.0e0) / Dimension::F;
                norm   = sqrt(norm);
                for (int i = 0; i < Dimension::F; ++i) c[i] /= norm;
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                norm = 0.0;
                for (int i = 0; i < Dimension::F; ++i) norm += std::abs(rho_ele[i * Dimension::Fadd1]);
                double gmeff = (norm - 1.0) / Dimension::F;
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), 1.0, gmeff, Dimension::F, use_cv, iocc);
                break;
            }
            case ElectronicSamplingPolicy::Gaussian: {
                // If c has been initialized as a normalized wavefunction, rho_ele has trace 1.
                // Otherwise rho_ele inherits the norm of c through rho_ele = |c><c|.
                // elec_utils::c_gaussian(c, Dimension::F); /// @debug
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), xi1, gamma1, Dimension::F, use_cv, iocc);
                w[0] = psnd_complex(Dimension::F);
                break;
            }
            case ElectronicSamplingPolicy::Constraint: {
                // c_sphere makes ||c|| = 1, so rho_ele = |c><c| has Tr(rho_ele) = 1.
                // rho_nuc is then the affine-mapped density with unit trace when use_cv=false.
                elec_utils::c_sphere(c.data(), Dimension::F);
                elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);
                elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), xi1, gamma1, Dimension::F, use_cv, iocc);
                w[0] = psnd_complex(Dimension::F);
                break;
            }
            case ElectronicSamplingPolicy::ReadDataSet: {  // @NOTE: or read from _dataset_load?
                std::string open_file = sampling_file;
                if (!isFileExists(sampling_file)) open_file = utils::concat(sampling_file, stat.icalc, ".ds");
                if (!isFileExists(open_file)) {
                    throw psnd_error(utils::concat("Sampling_Elec ReadDataSet cannot open electronic sampling file: ",
                                                   open_file));
                }
                std::string   eachline;
                std::ifstream ifs(open_file);
                if (!ifs.is_open()) {
                    throw psnd_error(utils::concat("Sampling_Elec ReadDataSet failed to open electronic sampling file: ",
                                                   open_file));
                }
                bool read_c       = false;
                bool read_rho_ele = false;
                bool read_rho_nuc = false;
                bool read_w       = false;
                while (getline(ifs, eachline)) {
                    eachline = trim_copy(eachline);
                    if (eachline == "init.c") {
                        if (!getline(ifs, eachline)) {
                            throw psnd_error(utils::concat("Sampling_Elec ReadDataSet malformed init.c in ", open_file));
                        }
                        for (int i = 0; i < Dimension::F; ++i) {
                            if (!(ifs >> c[i])) {
                                throw psnd_error(
                                    utils::concat("Sampling_Elec ReadDataSet failed to read init.c in ", open_file));
                            }
                        }
                        read_c = true;
                    }
                    if (eachline == "init.rho_ele") {
                        if (!getline(ifs, eachline)) {
                            throw psnd_error(
                                utils::concat("Sampling_Elec ReadDataSet malformed init.rho_ele in ", open_file));
                        }
                        for (int i = 0; i < Dimension::FF; ++i) {
                            if (!(ifs >> rho_ele[i])) {
                                throw psnd_error(utils::concat(
                                    "Sampling_Elec ReadDataSet failed to read init.rho_ele in ", open_file));
                            }
                        }
                        read_rho_ele = true;
                    }
                    if (eachline == "init.rho_nuc") {
                        if (!getline(ifs, eachline)) {
                            throw psnd_error(
                                utils::concat("Sampling_Elec ReadDataSet malformed init.rho_nuc in ", open_file));
                        }
                        for (int i = 0; i < Dimension::FF; ++i) {
                            if (!(ifs >> rho_nuc[i])) {
                                throw psnd_error(utils::concat(
                                    "Sampling_Elec ReadDataSet failed to read init.rho_nuc in ", open_file));
                            }
                        }
                        read_rho_nuc = true;
                    }
                    if (eachline == "init.w") {
                        if (!getline(ifs, eachline) || !(ifs >> w[0])) {
                            throw psnd_error(utils::concat("Sampling_Elec ReadDataSet failed to read init.w in ",
                                                           open_file));
                        }
                        read_w = true;
                    }
                }
                if (!read_c || !read_rho_ele || !read_rho_nuc || !read_w) {
                    std::vector<std::string> missing;
                    if (!read_c) missing.push_back("init.c");
                    if (!read_rho_ele) missing.push_back("init.rho_ele");
                    if (!read_rho_nuc) missing.push_back("init.rho_nuc");
                    if (!read_w) missing.push_back("init.w");
                    std::string missing_keys;
                    for (std::size_t i = 0; i < missing.size(); ++i) {
                        if (i > 0) missing_keys += ", ";
                        missing_keys += missing[i];
                    }
                    throw psnd_error(utils::concat("Sampling_Elec ReadDataSet missing electronic keys in ", open_file,
                                                   ": ", missing_keys,
                                                   ". Include electronic init data in init.ds or use another "
                                                   "solver.sampling_ele_flag."));
                }
                // elec_utils::ker_from_c(rho_ele.data(), c.data(), 1, 0, Dimension::F);  ///< initial rho_ele
                // elec_utils::ker_from_rho(rho_nuc.data(), rho_ele.data(), xi1, gamma1, Dimension::F, use_cv, iocc);
                // w[0] = phys::math::iu;
                break;
            }
        }

        // BO occupation: determine active state in adiabatic representation
        // See docs/dev/general_soc_representation.md for the occ/force bridging logic.
        if (Kernel_Representation::inp_repr_type == RepresentationPolicy::General_soc) {
            auto Tc = this->Tc.subspan(iP * Dimension::FF, Dimension::FF);
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                             RepresentationPolicy::General_soc,
                                             RepresentationPolicy::Adiabatic,
                                             SpacePolicy::L);
            occ_nuc[0] = elec_utils::max_choose(rho_nuc.data());
            if (use_fssh) occ_nuc[0] = elec_utils::pop_choose(rho_nuc.data());
            Kernel_Representation::transform(rho_nuc.data(), Tc.data(), Dimension::F,
                                             RepresentationPolicy::Adiabatic,
                                             RepresentationPolicy::General_soc,
                                             SpacePolicy::L);
        } else {
            Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,
                                             Kernel_Representation::inp_repr_type,
                                             Kernel_Representation::nuc_repr_type,
                                             SpacePolicy::L);
            occ_nuc[0] = elec_utils::max_choose(rho_nuc.data());
            if (use_fssh) occ_nuc[0] = elec_utils::pop_choose(rho_nuc.data());
            Kernel_Representation::transform(rho_nuc.data(), T.data(), Dimension::F,
                                             Kernel_Representation::nuc_repr_type,
                                             Kernel_Representation::inp_repr_type,
                                             SpacePolicy::L);
        }
    }
    _dataset->def(DATA::init::c, c);
    _dataset->def(DATA::init::rho_ele, rho_ele);
    _dataset->def(DATA::init::rho_nuc, rho_nuc);
    _dataset->def(DATA::init::T, T);
    _dataset->def(DATA::init::Tc, Tc);
    // _dataset->def(VARIABLE<psnd_complex>("init.c", &Dimension::shape_PF, "@"), c);
    // _dataset->def(VARIABLE<psnd_complex>("init.rho_ele", &Dimension::shape_PFF, "@"), rho_ele);
    // _dataset->def(VARIABLE<psnd_complex>("init.rho_nuc", &Dimension::shape_PFF, "@"), rho_nuc);
    // _dataset->def(VARIABLE<psnd_real>("init.T", &Dimension::shape_PFF, "@"), T);
    return stat;
}

// Status& Sampling_Elec::executeKernel_impl(Status& stat) { return stat; }

};  // namespace PROJECT_NS
