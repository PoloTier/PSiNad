#ifndef MODEL_SHARC_Interface_H
#define MODEL_SHARC_Interface_H

#include "psnd/Model.h"
#include "psnd/Policy.h"

#ifdef PSND_WITH_SHARC
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#endif

#include <set>

namespace PROJECT_NS {

// Full taxonomy from SHARC4 (see /home/lhc/sharc4/bin/SHARC_*.py).
// S0 only exercises ANALYTICAL and LVC; the rest are declared so they can be
// wired in per stage without re-touching the enum.
DEFINE_POLICY(SHARCBackend,
              // ab-initio (SHARC_ABINITIO subclasses, fork QC internally)
              ORCA, MOLCAS, MOLPRO, TURBOMOLE, BAGEL, GAUSSIAN,
              COLUMBUS, NWCHEM, PYSCF, AMS_ADF, MNDO, MOPACPI, BASICORCA,
              // SHARC_FAST subclasses (in-process, persistent)
              LVC, ANALYTICAL, SPAINN, SCHNARC,
              // composite
              QMMM, OPENMM, DROPLET, ECI,
              // auto-switching
              ADAPTIVE,
              NONE);

// pybind11's py::object has hidden (internal) visibility on GCC, so compiling
// Model_SHARC_Interface (default-visible since it is in a shared lib with
// -fvisibility=default) otherwise emits -Wattributes warnings for the py::object
// data members. GCC_VISIBILITY_PUSH_HIDDEN squashes them without changing ABI.
#if defined(PSND_WITH_SHARC) && defined(__GNUC__)
#pragma GCC visibility push(hidden)
#endif
class Model_SHARC_Interface final : public Model {
   public:
    virtual const std::string getName();

    virtual int getType() const;

    Model_SHARC_Interface() {};

    virtual ~Model_SHARC_Interface() {};

   private:
    SHARCBackend::_type backend;
    std::string         backend_name;    // matches SHARC module name, e.g. "SHARC_ANALYTICAL"
    std::string         template_file;
    std::string         resources_file;
    std::string         qmin_file;
    std::string         sharc_root;      // $SHARC; bin/ and lib/ added to sys.path
    std::set<std::string> features;      // h, grad, nacdr, soc, dm, phases, ...
    bool                has_mm;          // backend ∈ {QMMM, OPENMM, DROPLET, ECI}
    bool                has_soc;         // 'soc' feature requested
    bool                persistent;      // FAST subclasses are always persistent
    bool                keep_scratch;    // model.sharc_keep_scratch: true → -d/scratch; false → /tmp/psinad_sharc_XXXXXX (rm at finalize)
    std::string         scratch_dir;     // actual chosen scratchdir path (absolute); cleaned at finalize when !keep_scratch
    int                 natom;
    int                 nstates;         // from QM.in "states"

    // Scope: single process, single trajectory.
    // initializeKernel_impl is called once per Model instance, so a plain member
    // guarantees setup_interface() runs exactly once. Cross-trajectory caches
    // (FAST-ensemble) are explicitly out of scope — see
    // docs/dev/sharc_interface_plan.md § 12.
#ifdef PSND_WITH_SHARC
    pybind11::object  sharc_instance;
    pybind11::module_ numpy_mod;
    long              setup_interface_calls;   // must stay == 1
    double            coord_scale;             // PSiNad-Bohr → SHARC input unit (1/factor)
#endif

    // Common spans (same shape as Model_QMInterface)
    span<psnd_real> x, p;
    span<psnd_real> x0, p0;
    span<psnd_int>  atoms;
    span<psnd_real> mass;
    span<psnd_real> vpes, grad;
    span<psnd_real> V, dV;
    span<psnd_real> T, eig, dE;
    span<psnd_real> nac, nac_prev;
    span<psnd_real> dt_ptr, t_ptr;
    span<psnd_int>  istep_ptr;
    span<psnd_bint> succ_ptr, frez_ptr, last_attempt_ptr;
    span<psnd_int>  fail_type_ptr;

    // General_soc bindings. Model fills Vc (complex H with SOC) and dVc (complex
    // d/dR Vc). Real NAC is stored in the shared `nac` (model::rep::nac) above.
    // Kernel_Representation builds Ec/Tc and dEc = dVc + [nac, Vc] from these —
    // we only bind Ec/Tc/dEc here for inspection (record rules, tests).
    span<psnd_complex> Vc, dVc, Ec, Tc, dEc;

    virtual void    setInputParam_impl(std::shared_ptr<Param> PM);
    virtual void    setInputDataSet_impl(std::shared_ptr<DataSet> DS);
    virtual Status& initializeKernel_impl(Status& stat);
    virtual Status& executeKernel_impl(Status& stat);
    virtual Status& finalizeKernel_impl(Status& stat);
};
#if defined(PSND_WITH_SHARC) && defined(__GNUC__)
#pragma GCC visibility pop
#endif

};  // namespace PROJECT_NS
#endif  // MODEL_SHARC_Interface_H
