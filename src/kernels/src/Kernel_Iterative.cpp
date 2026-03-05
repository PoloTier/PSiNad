#include "psnd/Kernel_Iterative.h"

#include <cmath>

#include "psnd/hash_fnv1a.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

const std::string Kernel_Iterative::getName() { return "Kernel_Iterative"; }

int Kernel_Iterative::getType() const { return utils::hash(FUNCTION_NAME); }

void Kernel_Iterative::setInputParam_impl(std::shared_ptr<Param> PM) {
    t0    = _param->get_real({"model.t0", "solver.t0"}, LOC(), phys::time_d, 0.0);
    tend  = _param->get_real({"model.tend", "solver.tend"}, LOC(), phys::time_d, 1.0);
    dt0   = _param->get_real({"model.dt", "solver.dt"}, LOC(), phys::time_d, 0.1);
    sstep = _param->get_int({"solver.sstep"}, LOC(), 1);

    // set time grids
    const double block_size   = sstep * dt0;
    const double raw_blocks   = (tend - t0) / block_size;
    const double blocks_scale = (std::abs(raw_blocks) > 1.0) ? std::abs(raw_blocks) : 1.0;
    const double tol          = 1.0e-12 * blocks_scale;
    // keep floor-aligned stepping; tolerance only guards floating-point roundoff near integers
    int aligned_blocks = static_cast<int>(std::floor(raw_blocks + tol));
    if (aligned_blocks < 0) aligned_blocks = 0;

    nstep = sstep * aligned_blocks;
    nsamp = nstep / sstep + 1;

    // Dimension::sstep = sstep;
    // Dimension::nstep = nstep;
    // Dimension::nsamp = nsamp;
}

void Kernel_Iterative::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    t            = DS->def(DATA::control::t);
    dt           = DS->def(DATA::control::dt);
    istep        = DS->def(DATA::control::istep);
    isamp        = DS->def(DATA::control::isamp);
    at_condition = DS->def(DATA::control::at_condition);
    // save some variables
    DS->def(VARIABLE<psnd_int>("control.sstep", &Dimension::shape_1, "@"))[0] = sstep;
    DS->def(VARIABLE<psnd_int>("control.nstep", &Dimension::shape_1, "@"))[0] = nstep;
    DS->def(VARIABLE<psnd_int>("control.nsamp", &Dimension::shape_1, "@"))[0] = nsamp;
}

Status& Kernel_Iterative::initializeKernel_impl(Status& stat) {
    if (_param->get_string({"load", "solver.load"}, LOC(), "").find(":continue") != std::string::npos) {  //
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        // exactly copy from _dataset_load to _dataset
        // istep[0]          = _dataset_load->def_int("recover.istep", 1)[0]; // @TODO BUG
        // isamp[0]          = _dataset_load->def_int("recover.isamp", 1)[0]; // @TODO BUG
        stat.succ         = true;
        stat.last_attempt = false;
        stat.frozen       = false;
        stat.fail_type    = 0;
        return stat;
    }
    if (_param->get_string({"load", "solver.load"}, LOC(), "").find(":restart") != std::string::npos) {  //
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        t[0]              = _dataset_load->def_real("control.t", 1)[0];
        dt[0]             = dt0;
        isamp[0]          = 0;
        istep[0]          = 0;
        stat.succ         = true;
        stat.last_attempt = false;
        stat.frozen       = false;
        stat.fail_type    = 0;
        return stat;
    }
    t[0]              = t0;
    dt[0]             = dt0;
    istep[0]          = 0;
    isamp[0]          = 0;
    stat.succ         = true;
    stat.last_attempt = false;
    stat.frozen       = false;
    stat.fail_type    = 0;
    return stat;
}

Status& Kernel_Iterative::executeKernel_impl(Status& stat) {
    stat.first_step = true;
    while (istep[0] <= nstep) {
        if (istep[0] == nstep) {
            dt[0]       = 0;  // set dt=0 to remove dynamics! only record in last step
            stat.frozen = true;
        }
        at_condition[0] = (istep[0] % sstep == 0);
        isamp[0]        = istep[0] / sstep;
        for (auto& pkernel : _child_kernels) { pkernel->executeKernel(stat); }
        t[0] += dt[0];
        istep[0]++;
        stat.first_step = false;
    }
    return stat;
}
};  // namespace PROJECT_NS
