#include "psnd/Kernel_Load_DataSet.h"

// @blame thirdpart/filesystem is a wrapper of <filesystem> and compile with previous version of c++ (c++11)
#include <filesystem>
#include <algorithm>
#include <vector>

#include "psnd/hash_fnv1a.h"
#include "psnd/linalg.h"
#include "psnd/macro_utils.h"
#include "psnd/vars_list.h"

namespace PROJECT_NS {

namespace {

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool is_resume_mode(const std::string& load) { return load.find(":resume") != std::string::npos; }

bool is_record_leaf_for_resume(const std::string& key) {
    return starts_with(key, "record.") || starts_with(key, "_.0.record.") || starts_with(key, "_.1.record.") ||
           starts_with(key, "_.2.record.");
}

bool is_current_run_grid_leaf(const std::string& key) {
    return key == "control.nstep" || key == "control.nsamp" || key == "control.sstep" ||
           key == "control.msize" || key == "control.dt_backup" || key == "control.pertimeunit";
}

std::vector<std::size_t> shape_dims(Shape* shape) { return shape->dims(); }

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

void require_same_leaf_shape(const std::string& key, Shape* old_shape, Shape* new_shape) {
    const auto old_dims = shape_dims(old_shape);
    const auto new_dims = shape_dims(new_shape);
    if (old_dims != new_dims) {
        throw psnd_error(utils::concat("resume load shape mismatch for ", key, ": old ", old_shape->to_string(),
                                       ", current ", new_shape->to_string()));
    }
}

void copy_leaf_values(const std::string& key, psnd_dtype dtype, void* old_data, Shape* old_shape, void* new_data) {
    const int size = old_shape->size();
    switch (dtype) {
        case psnd_int_type:
            std::copy_n(static_cast<psnd_int*>(old_data), size, static_cast<psnd_int*>(new_data));
            break;
        case psnd_real_type:
            std::copy_n(static_cast<psnd_real*>(old_data), size, static_cast<psnd_real*>(new_data));
            break;
        case psnd_complex_type:
            std::copy_n(static_cast<psnd_complex*>(old_data), size, static_cast<psnd_complex*>(new_data));
            break;
        default:
            throw psnd_error(utils::concat("resume load unsupported dtype for ", key, ": ", enum_t_as_str(dtype)));
    }
}

void copy_resume_nonrecord_state(std::shared_ptr<DataSet>& from, std::shared_ptr<DataSet>& to) {
    std::vector<std::string> keys;
    collect_leaf_keys(from.get(), "", keys);

    for (const auto& key : keys) {
        if (is_record_leaf_for_resume(key)) continue;
        if (is_current_run_grid_leaf(key)) continue;

        psnd_dtype old_type;
        void*      old_data;
        Shape*     old_shape;
        std::tie(old_type, old_data, old_shape) = from->obtain(key);

        void* new_data = nullptr;
        if (to->haskey(key)) {
            psnd_dtype new_type;
            Shape*     new_shape;
            std::tie(new_type, new_data, new_shape) = to->obtain(key);
            if (old_type != new_type) {
                throw psnd_error(utils::concat("resume load dtype mismatch for ", key, ": old ",
                                               enum_t_as_str(old_type), ", current ", enum_t_as_str(new_type)));
            }
            require_same_leaf_shape(key, old_shape, new_shape);
        } else {
            switch (old_type) {
                case psnd_int_type:
                    new_data = to->def_int(key, *old_shape);
                    break;
                case psnd_real_type:
                    new_data = to->def_real(key, *old_shape);
                    break;
                case psnd_complex_type:
                    new_data = to->def_complex(key, *old_shape);
                    break;
                default:
                    throw psnd_error(
                        utils::concat("resume load unsupported dtype for ", key, ": ", enum_t_as_str(old_type)));
            }
        }
        copy_leaf_values(key, old_type, old_data, old_shape, new_data);
    }
}

}  // namespace

const std::string Kernel_Load_DataSet::getName() { return "Kernel_Load_DataSet"; }

int Kernel_Load_DataSet::getType() const { return utils::hash(FUNCTION_NAME); }

void Kernel_Load_DataSet::setInputParam_impl(std::shared_ptr<Param> PM) {
    load_fn = _param->get_string({"load", "solver.load"}, LOC(), "");
}

Status& Kernel_Load_DataSet::initializeKernel_impl(Status& stat) {
    if (load_fn == "") return stat;
    try {
        auto        ipos         = load_fn.find(":");
        std::string load_fn_file = (ipos == std::string::npos) ? load_fn : load_fn.substr(0, ipos);
        if (load_fn_file.find(".ds") != std::string::npos) {
            std::cout << "Loading DataSet from file: " << load_fn_file << "\n";
            // 先确认有没有这个文件 @ you can use ghc::filesystem instead for std=c++11
            if (!std::filesystem::exists(load_fn_file)) {
                throw psnd_error(utils::concat("File does not exist: ", load_fn_file));
            }

            if (load_fn_file.find(".bin.ds") != std::string::npos) {
                std::cout << "Loading dataset in binary format" << "\n";
                _dataset_load = std::shared_ptr<DataSet>(new DataSet());
                _dataset_load->load_binary(load_fn_file);
            } else {
                std::ifstream ifs{load_fn_file};

                if (!ifs.is_open()) { throw psnd_error(utils::concat("Cannot open file: ", load_fn_file)); }
                _dataset_load = std::shared_ptr<DataSet>(new DataSet());
                _dataset_load->load(ifs);
                // std::cout << _dataset_load->repr() << "\n";
                ifs.close();
            }

        } else {
            std::cout << "Loading DataSet from directory: "
                      << utils::concat(directory, "/", load_fn_file, stat.icalc, ".ds") << "\n";

            // 先确认有没有这个文件
            if (!std::filesystem::exists(utils::concat(directory, "/", load_fn_file, stat.icalc, ".ds"))) {
                throw std::runtime_error(utils::concat("File does not exist: ",
                                                       utils::concat(directory, "/", load_fn_file, stat.icalc, ".ds")));
            }

            std::ifstream ifs{utils::concat(directory, "/", load_fn_file, stat.icalc, ".ds")};
            if (!ifs.is_open()) {
                throw psnd_error(utils::concat("Cannot open file: ",
                                               utils::concat(directory, "/", load_fn_file, stat.icalc, ".ds")));
            }

            _dataset_load = std::shared_ptr<DataSet>(new DataSet());
            _dataset_load->load(ifs);

            ifs.close();
        }
        std::cout << "DataSet loaded successfully.\n";
        syncDataSetLoad(_dataset_load);
        std::cout << "DataSet synchronized successfully.\n";
        if (load_fn.find(":continue") != std::string::npos) {
            if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));

            std::cout << "Continuing DataSet from loaded state...\n";
            std::istringstream iss(_dataset_load->repr());
            _dataset->load(iss);
        } else if (load_fn.find(":restart") != std::string::npos) {
            if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));

            std::cout << "Reframing DataSet based on nsamp...\n";
            std::istringstream iss(_dataset_load->repr());
            int                nsamp = _dataset->def(VARIABLE<psnd_int>("control.nsamp", &Dimension::shape_1, "@"))[0];
            std::cout << "nsamp = " << nsamp << "\n";
            _dataset->load_reframe(iss, nsamp);
        } else if (is_resume_mode(load_fn)) {
            if (_dataset_load == nullptr) throw psnd_error(utils::concat(LOC(), ": DataSet Load error"));

            std::cout << "Resuming DataSet from loaded non-record state...\n";
            copy_resume_nonrecord_state(_dataset_load, _dataset);
        }
    } catch (const psnd_error& e) {
        // 重新抛出 psnd_error，保持友好的错误信息
        throw;
    } catch (const std::filesystem::filesystem_error& e) {
        throw psnd_error(
            utils::concat("❌ 文件系统错误: ", e.what(),
                          "\n请检查:\n  1. 文件路径是否正确\n  2. 文件权限是否足够\n  3. 磁盘空间是否充足"));
    } catch (const std::ios_base::failure& e) {
        throw psnd_error(
            utils::concat("❌ 文件读写错误: ", e.what(),
                          "\n请检查:\n  1. 文件是否存在且可读\n  2. 文件是否被其他程序占用\n  3. 磁盘是否有足够空间"));
    } catch (const std::runtime_error& e) {
        std::string original_error = e.what();
        throw psnd_error(
            utils::concat("❌ 数据集加载失败: ", original_error,
                          "\n这可能是由于:\n  1. 重启文件格式错误\n  2. 数据不完整或损坏\n  3. 配置参数不匹配"));
    } catch (const std::exception& e) {
        throw psnd_error(utils::concat("❌ 未知错误: ", e.what(),
            "\n如果问题持续出现，请联系开发团队并提供完整的错误信息"));
    } catch (...) {
        throw psnd_error("❌ 未知严重错误发生，程序终止。\n请联系开发团队并提供尽可能多的上下文信息以协助排查问题。");
    }
    return stat;
}

};  // namespace PROJECT_NS
