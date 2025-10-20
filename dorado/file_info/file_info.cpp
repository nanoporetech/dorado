#include "file_info/file_info.h"

#include "hts_utils/hts_types.h"
#include "utils/PostCondition.h"
#include "utils/fs_utils.h"
#include "utils/time_utils.h"

#include <pod5_format/c_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <stdexcept>

namespace dorado::file_info {

std::unordered_map<std::string, ReadGroup> load_read_groups(
        const std::vector<std::filesystem::directory_entry>& dir_files,
        const std::string& model_name,
        const std::string& modbase_model_names) {
    if (pod5_init() != POD5_OK) {
        throw std::runtime_error(
                fmt::format("Failed to initialise POD5: {}", pod5_get_error_string()));
    }

    std::unordered_map<std::string, ReadGroup> read_groups;
    for (const auto& entry : dir_files) {
        if (!utils::has_pod5_extension(entry)) {
            continue;
        }

        // Open the file
        const auto file_path = entry.path().string();
        Pod5FileReader_t* file = pod5_open_file(file_path.c_str());
        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_path, pod5_get_error_string());
            continue;
        }
        auto cleanup_file = utils::PostCondition([&file] {
            if (pod5_close_and_free_reader(file) != POD5_OK) {
                spdlog::error("Failed to close and free POD5 reader: {}", pod5_get_error_string());
            }
        });

        // First get the run info count
        run_info_index_t run_info_count = 0;
        if (pod5_get_file_run_info_count(file, &run_info_count) != POD5_OK) {
            spdlog::error("Failed to query run info count: {}", pod5_get_error_string());
            continue;
        }

        for (run_info_index_t idx = 0; idx < run_info_count; idx++) {
            RunInfoDictData_t* run_info_data = nullptr;
            if (pod5_get_file_run_info(file, idx, &run_info_data) != POD5_OK) {
                continue;
            }
            auto cleanup_run_info = utils::PostCondition([&run_info_data] {
                if (pod5_free_run_info(run_info_data) != POD5_OK) {
                    spdlog::error("Failed to free run info: {}", pod5_get_error_string());
                }
            });

            auto exp_start_time_ms = run_info_data->protocol_start_time_ms;
            std::string flowcell_id = run_info_data->flow_cell_id;
            std::string device_id = run_info_data->system_name;
            std::string run_id = run_info_data->protocol_run_id;
            std::string sample_id = run_info_data->sample_id;
            std::string position_id = run_info_data->sequencer_position;
            std::string experiment_id = run_info_data->experiment_name;

            std::string id = std::string(run_id).append("_").append(model_name);
            read_groups[id] = ReadGroup{
                    std::move(run_id),
                    model_name,
                    modbase_model_names,
                    std::move(flowcell_id),
                    std::move(device_id),
                    utils::get_string_timestamp_from_unix_time_ms(exp_start_time_ms),
                    std::move(sample_id),
                    std::move(position_id),
                    std::move(experiment_id),
            };
        }
    }

    return read_groups;
}

size_t get_num_reads(const std::vector<std::filesystem::directory_entry>& dir_files,
                     std::optional<std::unordered_set<std::string>> read_list,
                     const std::unordered_set<std::string>& ignore_read_list) {
    if (pod5_init() != POD5_OK) {
        throw std::runtime_error(
                fmt::format("Failed to initialise POD5: {}", pod5_get_error_string()));
    }

    size_t num_reads = 0;
    for (const auto& entry : dir_files) {
        const auto ext = utils::get_extension(entry);

        if (ext == ".fast5") {
            throw std::runtime_error("FAST5 is not supported");
        } else if (ext != ".pod5") {
            continue;
        }

        // Open the file
        const auto file_path = entry.path().string();
        Pod5FileReader_t* file = pod5_open_file(file_path.c_str());
        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_path, pod5_get_error_string());
            continue;
        }
        auto cleanup_file = utils::PostCondition([&file] {
            if (pod5_close_and_free_reader(file) != POD5_OK) {
                spdlog::error("Failed to close and free POD5 reader: {}", pod5_get_error_string());
            }
        });

        size_t read_count = 0;
        if (pod5_get_read_count(file, &read_count) != POD5_OK) {
            spdlog::error("Failed to query read count of {}: {}", file_path,
                          pod5_get_error_string());
            continue;
        }

        num_reads += read_count;
    }

    // Remove the reads in the ignore list from the total dataset read count.
    num_reads -= ignore_read_list.size();

    if (read_list) {
        // Get the unique read ids in the read list, since everything in the ignore
        // list will be skipped over.
        std::vector<std::string> final_read_list;
        std::set_difference(read_list->begin(), read_list->end(), ignore_read_list.begin(),
                            ignore_read_list.end(),
                            std::inserter(final_read_list, final_read_list.begin()));
        num_reads = std::min(num_reads, final_read_list.size());
    }

    return num_reads;
}

bool is_pod5_data_present(const std::vector<std::filesystem::directory_entry>& dir_files) {
    for (const auto& entry : dir_files) {
        if (utils::has_pod5_extension(entry)) {
            return true;
        }
    }
    return false;
}

uint16_t get_sample_rate(const std::vector<std::filesystem::directory_entry>& dir_files) {
    if (pod5_init() != POD5_OK) {
        throw std::runtime_error(
                fmt::format("Failed to initialise POD5: {}", pod5_get_error_string()));
    }

    for (const auto& entry : dir_files) {
        if (!utils::has_pod5_extension(entry)) {
            continue;
        }

        // Open the file
        auto file_path = entry.path().string();
        Pod5FileReader_t* file = pod5_open_file(file_path.c_str());
        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_path, pod5_get_error_string());
            continue;
        }
        auto cleanup_file = utils::PostCondition([&file, &file_path]() {
            if (pod5_close_and_free_reader(file) != POD5_OK) {
                spdlog::error("Failed to close and free POD5 reader for file {}: {}", file_path,
                              pod5_get_error_string());
            }
        });

        // First get the run info count
        run_info_index_t run_info_count = 0;
        if (pod5_get_file_run_info_count(file, &run_info_count) != POD5_OK) {
            spdlog::error("Failed to fetch POD5 run info count for file {} : {}", file_path,
                          pod5_get_error_string());
            continue;
        }
        if (run_info_count > static_cast<run_info_index_t>(0)) {
            RunInfoDictData_t* run_info_data = nullptr;
            if (pod5_get_file_run_info(file, 0, &run_info_data) != POD5_OK) {
                spdlog::error(
                        "Failed to fetch POD5 run info dict for file {} and run info index 0: {}",
                        file_path, pod5_get_error_string());
                continue;
            }
            auto cleanup_run_info = utils::PostCondition([&run_info_data, &file_path] {
                if (pod5_free_run_info(run_info_data) != POD5_OK) {
                    spdlog::error(
                            "Failed to free POD5 run info for file {} and run info index 0: {}",
                            file_path, pod5_get_error_string());
                }
            });

            // Break out of loop if sample rate is found.
            return run_info_data->sample_rate;
        }
    }

    throw std::runtime_error("Unable to determine sample rate for data.");
}

static std::set<models::ChemistryKey> get_sequencing_chemistries(
        const std::vector<std::filesystem::directory_entry>& dir_files) {
    if (pod5_init() != POD5_OK) {
        throw std::runtime_error(
                fmt::format("Failed to initialise POD5: {}", pod5_get_error_string()));
    }

    std::set<models::ChemistryKey> chemistries;
    for (const auto& entry : dir_files) {
        if (!utils::has_pod5_extension(entry)) {
            continue;
        }

        const auto file_path = std::filesystem::path(entry).string();

        // Open the file
        Pod5FileReader_t* file = pod5_open_file(file_path.c_str());
        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_path, pod5_get_error_string());
            continue;
        }
        auto cleanup_file = utils::PostCondition([&file, &file_path] {
            if (pod5_close_and_free_reader(file) != POD5_OK) {
                spdlog::error("Failed to close and free POD5 reader for file {}: {}", file_path,
                              pod5_get_error_string());
            }
        });

        // First get the run info count
        run_info_index_t run_info_count = 0;
        if (pod5_get_file_run_info_count(file, &run_info_count) != POD5_OK) {
            spdlog::error("Failed to fetch POD5 run info count for file {} : {}", file_path,
                          pod5_get_error_string());
            continue;
        }

        for (run_info_index_t ri_idx = 0; ri_idx < run_info_count; ri_idx++) {
            RunInfoDictData_t* run_info_data = nullptr;
            if (pod5_get_file_run_info(file, ri_idx, &run_info_data) != POD5_OK) {
                spdlog::error(
                        "Failed to fetch POD5 run info dict for file {} and run info index {}: {}",
                        file_path, ri_idx, pod5_get_error_string());
                continue;
            }
            auto cleanup_run_info = utils::PostCondition([&run_info_data, &file_path] {
                if (pod5_free_run_info(run_info_data) != POD5_OK) {
                    spdlog::error("Failed to free POD5 run info for file {}: {}", file_path,
                                  pod5_get_error_string());
                }
            });

            const auto chemistry_key = models::get_chemistry_key(
                    run_info_data->flow_cell_product_code, run_info_data->sequencing_kit,
                    run_info_data->sample_rate);
            spdlog::trace("POD5: {} {}", file_path, to_string(chemistry_key));
            chemistries.insert(chemistry_key);
        }
    }

    return chemistries;
}

models::Chemistry get_unique_sequencing_chemistry(
        const std::vector<std::filesystem::directory_entry>& dir_files) {
    std::set<models::ChemistryKey> data_chemistries = get_sequencing_chemistries(dir_files);

    if (data_chemistries.empty()) {
        throw std::runtime_error(
                "Failed to determine sequencing chemistry from data. Please select a model by "
                "path");
    }

    std::set<models::Chemistry> found;
    for (const auto& dc : data_chemistries) {
        const auto chemistry = models::get_chemistry(dc);
        if (chemistry == models::Chemistry::UNKNOWN) {
            spdlog::error("No supported chemistry found for {}", to_string(dc));
            spdlog::error(
                    "This is typically seen when using prototype kits. Please download an "
                    "appropriate model for your data and select it by model path");

            throw std::runtime_error("Could not resolve chemistry from data: Unknown chemistry");
        }
        found.insert(chemistry);
    }
    if (found.empty()) {
        throw std::runtime_error("Could not resolve chemistry from data: No data");
    }
    if (found.size() > 1) {
        spdlog::error("Multiple sequencing chemistries found in data");
        for (auto f : found) {
            spdlog::error("Found: {}", to_string(f));
        }

        throw std::runtime_error("Could not uniquely resolve chemistry from inhomogeneous data");
    }
    return *std::begin(found);
}

}  // namespace dorado::file_info
