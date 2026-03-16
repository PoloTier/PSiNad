#include "psnd/Kernel.h"
#include "psnd/Kernel_Dump_DataSet.h"
#include "psnd/Kernel_Elec_Functions.h"
#include "psnd/Kernel_Load_DataSet.h"
#include "psnd/Kernel_NAForce.h"
#include "psnd/Kernel_Random.h"
#include "psnd/Kernel_Read_Dimensions.h"
#include "psnd/Kernel_Representation.h"
#include "psnd/Model.h"
#include "psnd/Sampling_Elec.h"
#include "psnd/Sampling_Nucl.h"
#include "psnd/Solver.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

namespace PROJECT_NS {

EXPORT std::shared_ptr<Solver> Sampling_Kernel(std::shared_ptr<Model> kmodel, std::string Kernel_name) {
    // Root Kernel
    std::shared_ptr<Kernel> ker(new Kernel(Kernel_name));

    std::shared_ptr<Kernel_Representation> krepr(new Kernel_Representation());
    std::shared_ptr<Kernel_NAForce>        knaf(new Kernel_NAForce());
    std::shared_ptr<Kernel_Elec_Functions> kfuncs(new Kernel_Elec_Functions());

    // Order matters:
    // - kmodel must run before krepr to produce V/dV for T/eig/E/dE/H.
    // - Sampling_Elec needs T (from krepr) to set occ_nuc and rho in target repr.
    // - kfuncs/knaf consume T, rho_ele/rho_nuc, occ_nuc (so must run after Sampling_Elec).
    ker->appendChild(std::shared_ptr<Kernel_Load_DataSet>(new Kernel_Load_DataSet()))
        .appendChild(std::shared_ptr<Kernel_Random>(new Kernel_Random()))
        .appendChild(std::shared_ptr<Kernel_Read_Dimensions>(new Kernel_Read_Dimensions()))
        .appendChild(std::shared_ptr<Sampling_Nucl>(new Sampling_Nucl()))
        .appendChild(kmodel)
        .appendChild(krepr)
        .appendChild(std::shared_ptr<Sampling_Elec>(new Sampling_Elec()))
        .appendChild(kfuncs)
        .appendChild(knaf)
        .appendChild(std::shared_ptr<Kernel_Dump_DataSet>(new Kernel_Dump_DataSet()));
    std::shared_ptr<Solver> sol(new Solver(ker));
    return sol;
}

};  // namespace PROJECT_NS
