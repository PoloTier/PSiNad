#include "psnd/Kernel_Iterative.h"

#include <algorithm>
#include <cmath>

#include "psnd/hash_fnv1a.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

namespace {

bool is_resume_mode(const std::string& load) { return load.find(":resume") != std::string::npos; }

bool is_loaded_trajectory_mode(const std::string& load) {
    return load.find(":continue") != std::string::npos || load.find(":restart") != std::string::npos ||
           is_resume_mode(load);
}

psnd_int get_loaded_int(std::shared_ptr<DataSet>& dataset, const std::string& key) {
    psnd_dtype dtype;
    void*      data;
    Shape*     shape;
    std::tie(dtype, data, shape) = dataset->obtain(key);
    if (dtype != psnd_int_type || shape->size() != 1) {
        throw psnd_error(utils::concat("resume requires scalar int key ", key));
    }
    return static_cast<psnd_int*>(data)[0];
}

psnd_real get_loaded_real(std::shared_ptr<DataSet>& dataset, const std::string& key) {
    psnd_dtype dtype;
    void*      data;
    Shape*     shape;
    std::tie(dtype, data, shape) = dataset->obtain(key);
    if (dtype != psnd_real_type || shape->size() != 1) {
        throw psnd_error(utils::concat("resume requires scalar real key ", key));
    }
    return static_cast<psnd_real*>(data)[0];
}

void check_resume_grid(std::shared_ptr<DataSet>& dataset, int current_sstep, double current_dt0) {
    const int old_sstep = get_loaded_int(dataset, "control.sstep");
    if (old_sstep != current_sstep) {
        throw psnd_error(utils::concat("resume requires unchanged solver.sstep: old ", old_sstep, ", current ",
                                       current_sstep));
    }
    if (dataset->haskey("control.dt")) {
        const double old_dt = get_loaded_real(dataset, "control.dt");
        const double scale  = std::max({1.0, std::abs(old_dt), std::abs(current_dt0)});
        if (std::abs(old_dt) > 1.0e-14 && std::abs(old_dt - current_dt0) > 1.0e-10 * scale) {
            throw psnd_error(utils::concat("resume requires compatible timestep: old control.dt ", old_dt,
                                           ", current dt ", current_dt0));
        }
    }
}

void check_resume_time_grid(int old_istep, double old_t, double current_t0, double current_dt0) {
    const double expected_t = current_t0 + old_istep * current_dt0;
    const double scale      = std::max({1.0, std::abs(old_t), std::abs(expected_t)});
    if (std::abs(old_t - expected_t) > 1.0e-10 * scale) {
        throw psnd_error(utils::concat("resume requires compatible time grid: loaded control.istep/control.t imply dt ",
                                       (old_istep == 0 ? 0.0 : (old_t - current_t0) / old_istep),
                                       ", current dt ", current_dt0));
    }
}

}  // namespace

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
    const std::string load_str = _param->get_string({"load", "solver.load"}, LOC(), "");
    if (load_str.find(":continue") != std::string::npos) {  //
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        // exactly copy from _dataset_load to _dataset
        // istep[0]          = _dataset_load->def_int("recover.istep", 1)[0]; // @TODO BUG
        // isamp[0]          = _dataset_load->def_int("recover.isamp", 1)[0]; // @TODO BUG
        stat.succ         = true;
        stat.last_attempt = false;
        stat.first_step   = false;
        stat.frozen       = false;
        stat.fail_type    = 0;
        return stat;
    }
    if (load_str.find(":restart") != std::string::npos) {  //
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        t[0]              = _dataset_load->def_real("control.t", 1)[0];
        dt[0]             = dt0;
        isamp[0]          = 0;
        istep[0]          = 0;
        stat.succ         = true;
        stat.last_attempt = false;
        stat.first_step   = false;
        stat.frozen       = false;
        stat.fail_type    = 0;
        return stat;
    }
    if (is_resume_mode(load_str)) {
        if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));
        check_resume_grid(_dataset_load, sstep, dt0);

        const int    old_istep = get_loaded_int(_dataset_load, "control.istep");
        const double old_t     = get_loaded_real(_dataset_load, "control.t");
        check_resume_time_grid(old_istep, old_t, t0, dt0);

        istep[0]          = old_istep;
        t[0]              = old_t;
        dt[0]             = dt0;
        isamp[0]          = istep[0] / sstep;
        stat.succ         = true;
        stat.last_attempt = false;
        stat.first_step   = false;
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
    stat.first_step   = true;
    stat.frozen       = false;
    stat.fail_type    = 0;
    return stat;
}

Status& Kernel_Iterative::executeKernel_impl(Status& stat) {
    const std::string load_str = _param->get_string({"load", "solver.load"}, LOC(), "");
    stat.first_step            = !is_loaded_trajectory_mode(load_str);
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
