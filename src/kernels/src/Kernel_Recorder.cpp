#include "psnd/Kernel_Recorder.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "psnd/Einsum.h"
#include "psnd/RuleEvaluator.h"
#include "psnd/RuleSet.h"
#include "psnd/chem.h"
#include "psnd/debug_utils.h"
#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/phys.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool is_resume_mode(const std::string& load) { return load.find(":resume") != std::string::npos; }

void collect_leaf_keys(DataSet* dataset, const std::string& prefix, std::vector<std::string>& keys) {
    for (auto& item : *(dataset->_data)) {
        if (!item.second) continue;
        const std::string key = prefix.empty() ? item.first : utils::concat(prefix, ".", item.first);
        Node*             node = item.second.get();
        if (node->type() == psnd_dataset_type) {
            collect_leaf_keys(static_cast<DataSet*>(node), key, keys);
        } else {
            keys.push_back(key);
        }
    }
}

std::vector<std::size_t> shape_dims(Shape* shape) { return shape->dims(); }

std::size_t per_frame_size(const std::vector<std::size_t>& dims) {
    std::size_t size = 1;
    for (std::size_t i = 1; i < dims.size(); ++i) size *= dims[i];
    return size;
}

void copy_record_prefix_values(const std::string& key, psnd_dtype dtype, void* old_data, void* new_data,
                               std::size_t count) {
    switch (dtype) {
        case psnd_real_type:
            std::copy_n(static_cast<psnd_real*>(old_data), count, static_cast<psnd_real*>(new_data));
            break;
        case psnd_complex_type:
            std::copy_n(static_cast<psnd_complex*>(old_data), count, static_cast<psnd_complex*>(new_data));
            break;
        case psnd_int_type:
            std::copy_n(static_cast<psnd_int*>(old_data), count, static_cast<psnd_int*>(new_data));
            break;
        default:
            throw psnd_error(utils::concat("resume record unsupported dtype for ", key, ": ", enum_t_as_str(dtype)));
    }
}

void import_record_key(std::shared_ptr<DataSet>& old_dataset, std::shared_ptr<DataSet>& new_dataset,
                       const std::string& key, int resume_frame, bool required) {
    if (!new_dataset->haskey(key)) return;
    if (!old_dataset->haskey(key)) {
        if (required) throw psnd_error(utils::concat("resume record missing old key: ", key));
        return;
    }

    psnd_dtype old_type;
    psnd_dtype new_type;
    void*      old_data;
    void*      new_data;
    Shape*     old_shape;
    Shape*     new_shape;
    std::tie(old_type, old_data, old_shape) = old_dataset->obtain(key);
    std::tie(new_type, new_data, new_shape) = new_dataset->obtain(key);

    if (old_type != new_type) {
        throw psnd_error(utils::concat("resume record dtype mismatch for ", key, ": old ", enum_t_as_str(old_type),
                                       ", current ", enum_t_as_str(new_type)));
    }

    const auto old_dims = shape_dims(old_shape);
    const auto new_dims = shape_dims(new_shape);
    if (old_dims.size() != new_dims.size() || old_dims.empty()) {
        throw psnd_error(utils::concat("resume record rank mismatch for ", key));
    }
    for (std::size_t i = 1; i < old_dims.size(); ++i) {
        if (old_dims[i] != new_dims[i]) {
            throw psnd_error(utils::concat("resume record per-frame shape mismatch for ", key, ": old ",
                                           old_shape->to_string(), ", current ", new_shape->to_string()));
        }
    }

    const std::size_t old_frames  = old_dims[0];
    const std::size_t new_frames  = new_dims[0];
    const std::size_t frame_limit = static_cast<std::size_t>(std::max(0, resume_frame) + 1);
    const std::size_t copy_frames = std::min({old_frames, new_frames, frame_limit});
    const std::size_t count       = copy_frames * per_frame_size(old_dims);
    copy_record_prefix_values(key, old_type, old_data, new_data, count);
}

void import_resume_record_history(std::shared_ptr<DataSet>& old_dataset, std::shared_ptr<DataSet>& new_dataset,
                                  int resume_frame) {
    if (old_dataset == nullptr) throw psnd_error("resume record import requires loaded DataSet");

    std::vector<std::string> keys;
    collect_leaf_keys(new_dataset.get(), "", keys);
    for (const auto& key0 : keys) {
        if (!starts_with(key0, "_.0.record.")) continue;
        import_record_key(old_dataset, new_dataset, key0, resume_frame, true);

        const std::string suffix = key0.substr(std::string("_.0.record.").size());
        import_record_key(old_dataset, new_dataset, utils::concat("_.1.record.", suffix), resume_frame, false);
        import_record_key(old_dataset, new_dataset, utils::concat("_.2.record.", suffix), resume_frame, false);
    }
}

}  // namespace

const std::string Kernel_Recorder::getName() { return "Kernel_Recorder"; }

int Kernel_Recorder::getType() const { return utils::hash(FUNCTION_NAME); }

Kernel_Recorder::Kernel_Recorder() {
    _ruleset = std::shared_ptr<RuleSet>(new RuleSet());  //
}

Kernel_Recorder::~Kernel_Recorder() {};

void Kernel_Recorder::setInputParam_impl(std::shared_ptr<Param> PM) {
    dt              = _param->get_real({"model.dt", "solver.dt"}, LOC(), phys::time_d);
    t0              = _param->get_real({"model.t0", "solver.t0"}, LOC(), phys::time_d, 0.0);
    time_unit       = _param->get_real({"model.time_unit", "solver.time_unit"}, LOC(), phys::time_d, 1.0);
    occ0            = _param->get_int({"model.occ", "solver.occ"}, LOC(), -1);
    record_dumpstep = _param->get_int({"solver.record_dumpstep"}, LOC(), 0);
    record_tmp      = _param->get_bool({"solver.record_tmp"}, LOC(), false);
    record_xyz      = _param->get_bool({"solver.record_xyz"}, LOC(), false);
}

void Kernel_Recorder::setInputDataSet_impl(std::shared_ptr<DataSet> DS) {
    istep_ptr = DS->def(DATA::control::istep);
    sstep_ptr = DS->def(DATA::control::sstep);
    isamp_ptr = DS->def(DATA::control::isamp);
    nsamp_ptr = DS->def(DATA::control::nsamp);
    // set time unit in recorder
    DS->def(DATA::control::pertimeunit)[0] = 1.0e0 / time_unit;

    if (record_xyz) {
        atoms_ptr = DS->def(DATA::model::atoms);
        x_ptr     = DS->def(DATA::integrator::x);
        natom     = static_cast<int>(Dimension::N / 3);
    }
}

void Kernel_Recorder::writeXYZFrame(std::ofstream& ofs, const psnd_real* x_data, const std::string& comment) {
    ofs << natom << "\n";
    ofs << comment << "\n";
    for (int i = 0, idx = 0; i < natom; ++i) {
        ofs << FMT(8) << chem::getElemLabel(atoms_ptr[i])  //
            << FMT(8) << x_data[idx++] * phys::au_2_ang    //
            << FMT(8) << x_data[idx++] * phys::au_2_ang    //
            << FMT(8) << x_data[idx++] * phys::au_2_ang << "\n";
    }
}

void Kernel_Recorder::parse() {
    if (!_param->is_array("record")) return;

    int it = 0;
    while (true) {
        std::string firstkey = utils::concat("record.", it);
        if (!_param->has_key(firstkey)) break;

        std::string rule = "", mode = "average", save = "res.dat";

        if (false) {
        } else if (_param->is_object(firstkey)) {
            std::string mode_key = utils::concat(firstkey, ".", "mode");
            std::string save_key = utils::concat(firstkey, ".", "save");
            std::string rule_key = utils::concat(firstkey, ".", "rule");
            if (_param->has_key(mode_key)) mode = _param->get_string({mode_key}, LOC());
            if (_param->has_key(save_key)) save = _param->get_string({save_key}, LOC());
            if (_param->has_key(rule_key)) {
                rule = _param->get_string({rule_key}, LOC());
            } else {
                throw psnd_error("parser rule error");
            }
        } else if (_param->is_array(firstkey)) {
            std::string firstv0key = utils::concat(firstkey, ".", "0");
            std::string firstv1key = utils::concat(firstkey, ".", "1");
            std::string v0, v1;
            if (_param->has_key(firstv0key)) v0 = _param->get_string({firstv0key}, LOC());
            if (_param->has_key(firstv1key)) v1 = _param->get_string({firstv1key}, LOC());

            auto ipos0 = v0.find_first_of("#");
            if (ipos0 != std::string::npos) v0 = v0.substr(0, ipos0);
            auto ipos1 = v1.find_first_of("#");
            if (ipos1 != std::string::npos) v1 = v1.substr(0, ipos1);
            rule = utils::concat("_", it, "(", v0, ",", v1, ")");
        } else if (_param->is_string(firstkey)) {
            rule = _param->get_string({firstkey}, LOC());
        } else {
            throw psnd_error("unknown type");
        }

        if (std::find(opened_files.begin(), opened_files.end(), save) == opened_files.end() && mode == "average") {
            std::shared_ptr<RuleEvaluator> record_time_rule(  //
                new RuleEvaluator("time(t{control}:R, pertimeunit{control}:R)", _dataset, mode, save, nsamp_ptr[0]));
            _ruleset->registerRules(record_time_rule);
            opened_files.push_back(save);
        }

        if (occ0 >= 0) {  // replace some env in rules
            std::string            dst_str = rule;
            std::string            sub_str = "[occ]";
            std::string            new_str = utils::concat("[", occ0, "]");
            std::string::size_type pos     = 0;
            while ((pos = dst_str.find(sub_str)) != std::string::npos) {
                dst_str.replace(pos, sub_str.length(), new_str);
            }
            rule = dst_str;
        }

        std::shared_ptr<RuleEvaluator> record_rule(  //
            new RuleEvaluator(rule, _dataset, mode, save, nsamp_ptr[0]));
        _ruleset->registerRules(record_rule);

        it++;
    }
}

Status& Kernel_Recorder::initializeKernel_impl(Status& stat) {
    bool not_parsed = _ruleset->getRules().size() == 0;
    if (not_parsed) parse();

    const std::string load_str = _param->get_string({"load", "solver.load"}, LOC(), "");
    if (is_resume_mode(load_str)) import_resume_record_history(_dataset_load, _dataset, isamp_ptr[0]);

    if (record_xyz) {
        for (int i = 0; i < natom; ++i) {
            if (atoms_ptr[i] <= 0) {
                throw psnd_error(utils::concat(
                    "record_xyz=true requires a model with real atomic identities, ",
                    "but atoms[", i, "]=", atoms_ptr[i], " is not a valid Z."));
            }
        }
        // Continue/restart semantics for traj.xyz.
        //
        // Iteration order in NAD / NAD-adaptM is `Recorder -> Integrator -> SHARC`
        // (see NAD_AdaptM_Kernel.cpp), so when a Recorder dump fires inside
        // iteration K, traj.xyz already contains an `isamp=K` frame BEFORE
        // SHARC has been called for step K. SHARC's STEP file is updated
        // atomically in Model_SHARC_Interface (write_step_file runs only after
        // sharc.run() succeeds and the wrapper does not catch exceptions), so
        // on disk after a crash:
        //   * SAVE/STEP == K-1 if SHARC step K crashed (or never started)
        //   * SAVE/STEP == K   if SHARC step K succeeded and a later kernel crashed
        // In either case the latest record-dump on disk has control.istep == K.
        //
        // On continue, PSiNad sends "step K" to SHARC. SHARC's _step_logic
        // accepts both `K == last+1` (newstep -> fresh QM with last step's
        // orbitals as initial guess) and `K == last` (samestep -> redo step K),
        // so neither crash mode produces a step-number conflict. (A mismatch
        // only happens if you pick an old dump from a fully completed run
        // where SAVE has been advanced far past dump.istep — not a real-crash
        // workflow.)
        //
        // Because frame K is already in traj.xyz from the pre-crash Recorder
        // call, the resumed iteration K must skip its FIRST Recorder append to
        // avoid a duplicate seam frame. That is the sole purpose of
        // `skip_first_xyz_append`.
        //
        // Cross-directory continue is supported but the user must `cp
        // seed_dir/traj.xyz new_dir/` themselves before running — we do not
        // replay history from bin.ds, because doing so would duplicate the
        // user's own `integrator.x` recording rule (if any) inside the dump.
        const bool        is_load  = load_str.find(":restart") != std::string::npos  //
                                  || load_str.find(":continue") != std::string::npos  //
                                  || is_resume_mode(load_str);
        if (is_load) {
            skip_first_xyz_append = true;
        } else {
            std::ofstream ofs(utils::concat(this->directory, "/traj.xyz"), std::ios::trunc);
            ofs.close();
        }
    }
    return stat;
}

Status& Kernel_Recorder::executeKernel_impl(Status& stat) {
    if (_param->get_string({"load", "solver.load"}, LOC(), "").find(":restart") != std::string::npos) {
        for (auto& irule : _ruleset->getRules()) { irule->calculateResult(isamp_ptr[0], false); }
    } else {
        for (auto& irule : _ruleset->getRules()) { irule->calculateResult(isamp_ptr[0], true); }
    }
    if (record_xyz) {
        if (skip_first_xyz_append) {
            skip_first_xyz_append = false;
        } else {
            std::ofstream ofs(utils::concat(this->directory, "/traj.xyz"), std::ios::app);
            writeXYZFrame(ofs, x_ptr.data(),
                          utils::concat("isamp=", isamp_ptr[0], " istep=", istep_ptr[0]));
            ofs.close();
        }
    }
    if (record_tmp) RuleSet::flush_all(this->directory, ".TMP", 0);
    if (record_dumpstep > 0) {
        try {
            std::ofstream ofs{utils::concat(directory, "/record-dump", stat.icalc, "-", istep_ptr[0], ".ds")};
            _dataset->dump(ofs);
            ofs.close();
        } catch (std::runtime_error& e) { throw psnd_error("bad dump in recording\n"); }
    }
    return stat;
}

Status& Kernel_Recorder::finalizeKernel_impl(Status& stat) {
    for (auto& irule : _ruleset->getRules()) irule->collectResult();
    return stat;
}

};  // namespace PROJECT_NS
