#include "DataLoader.h"

#include "models/kits.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/messages.h"
#include "utils/PostCondition.h"
#include "utils/fs_utils.h"
#include "utils/thread_naming.h"
#include "utils/time_utils.h"
#include "utils/types.h"
#include "vbz_plugin_user_utils.h"

#include <ATen/Functions.h>
#include <cxxpool.h>
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <pod5_format/c_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dorado {

namespace {

// ReadID should be a drop-in replacement for read_id_t
static_assert(sizeof(dorado::ReadID) == sizeof(read_id_t));

// 37 = number of bytes in UUID (32 hex digits + 4 dashes + null terminator)
const uint32_t POD5_READ_ID_LEN = 37;

void string_reader(HighFive::Attribute& attribute, std::string& target_str) {
    // Load as a variable string if possible
    if (attribute.getDataType().isVariableStr()) {
        attribute.read(target_str);
        return;
    }

    // Process as a fixed length string
    // Create landing buffer and H5 datatype
    size_t size = attribute.getDataType().getSize();
    std::vector<char> target_array(size);
    hid_t dtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(dtype, size);

    // Copy to landing buffer
    if (H5Aread(attribute.getId(), dtype, target_array.data()) < 0) {
        throw std::runtime_error("Error during H5Aread of fixed length string");
    }

    // Extract to string
    target_str = std::string(target_array.data(), size);
    // It's possible the null terminator appears before the end of the string
    size_t eol_pos = target_str.find(char(0));
    if (eol_pos < target_str.size()) {
        target_str.resize(eol_pos);
    }
}

std::string get_string_attribute(const HighFive::Group& group, const std::string& attr_name) {
    std::string attribute_string;
    if (group.hasAttribute(attr_name)) {
        HighFive::Attribute attribute = group.getAttribute(attr_name);
        string_reader(attribute, attribute_string);
    }
    return attribute_string;
}

std::optional<std::vector<std::filesystem::directory_entry>> filter_fast5_for_mixed_datasets(
        const std::vector<std::filesystem::directory_entry>& files) {
    std::vector<std::filesystem::directory_entry> pod5_entries;
    std::vector<std::filesystem::directory_entry> fast5_entries;

    for (const auto& entry : files) {
        auto entry_path = std::filesystem::path(entry);
        std::string ext = entry_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".fast5") {
            fast5_entries.push_back(entry);
        } else if (ext == ".pod5") {
            pod5_entries.push_back(entry);
        }
    }

    if (!pod5_entries.empty() && !fast5_entries.empty()) {
        spdlog::error(
                "Data folder contains both POD5 and FAST5 files. FAST5 loading is unoptimized and "
                "will result in poor performance. Please basecall FAST5 separately or convert your "
                "dataset to POD5: https://pod5-file-format.readthedocs.io/en/latest/docs/"
                "tools.html#pod5-convert-fast5");
        // Note: once we drop POD5 support we can remove the use of optional here and in the caller
        return std::nullopt;
    }

    if (!fast5_entries.empty()) {
        spdlog::warn(
                "Deprecation Warning: FAST5 support in Dorado will be dropped in an "
                "upcoming release. FAST5 loading is unoptimized and will result in poor "
                "performance. Please convert your dataset to POD5: "
                "https://pod5-file-format.readthedocs.io/en/latest/docs/"
                "tools.html#pod5-convert-fast5");
        return fast5_entries;
    }

    return pod5_entries;
}

// Parses pod5 run_info data into a ChemistryKey which is used to lookup the sequencing chemistry
models::ChemistryKey get_chemistry_key(const RunInfoDictData_t* const run_info_data) {
    return models::get_chemistry_key(run_info_data->flow_cell_product_code,
                                     run_info_data->sequencing_kit, run_info_data->sample_rate);
}

SimplexReadPtr process_pod5_thread_fn(
        size_t row,
        Pod5ReadRecordBatch* batch,
        Pod5FileReader* file,
        const std::string& path,
        const std::unordered_map<int, std::vector<DataLoader::ReadSortInfo>>& reads_by_channel,
        const std::unordered_map<std::string, size_t>& read_id_to_index) {
    utils::set_thread_name("process_pod5");
    uint16_t read_table_version = 0;
    ReadBatchRowInfo_t read_data;
    if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                          &read_table_version) != POD5_OK) {
        spdlog::error("Failed to get read {}", row);
    }

    //Retrieve global information for the run
    RunInfoDictData_t* run_info_data;
    if (pod5_get_run_info(batch, read_data.run_info, &run_info_data) != POD5_OK) {
        spdlog::error("Failed to get Run Info {}{}", row, pod5_get_error_string());
    }
    auto run_acquisition_start_time_ms = run_info_data->acquisition_start_time_ms;
    auto run_sample_rate = run_info_data->sample_rate;

    char read_id_tmp[POD5_READ_ID_LEN];
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id");
    }
    std::string read_id_str(read_id_tmp);

    auto options = at::TensorOptions().dtype(at::kShort);
    auto samples = at::empty(read_data.num_samples, options);

    if (pod5_get_read_complete_signal(file, batch, row, read_data.num_samples,
                                      samples.data_ptr<int16_t>()) != POD5_OK) {
        spdlog::error("Failed to get read {} signal: {}", row, pod5_get_error_string());
    }

    auto new_read = std::make_unique<SimplexRead>();
    new_read->read_common.raw_data = samples;
    new_read->read_common.sample_rate = run_sample_rate;

    auto start_time_ms = run_acquisition_start_time_ms +
                         ((read_data.start_sample * 1000) /
                          (uint64_t)run_sample_rate);  // TODO check if this cast is needed
    auto start_time = utils::get_string_timestamp_from_unix_time(start_time_ms);
    new_read->run_acquisition_start_time_ms = run_acquisition_start_time_ms;
    new_read->read_common.start_time_ms = start_time_ms;
    new_read->scaling = read_data.calibration_scale;
    new_read->offset = read_data.calibration_offset;
    new_read->read_common.read_id = std::move(read_id_str);
    new_read->read_common.num_trimmed_samples = 0;
    new_read->read_common.attributes.read_number = read_data.read_number;
    new_read->read_common.attributes.filename = std::filesystem::path(path).filename().string();
    new_read->read_common.attributes.mux = read_data.well;
    new_read->read_common.attributes.num_samples = read_data.num_samples;
    new_read->read_common.attributes.channel_number = read_data.channel;
    new_read->read_common.attributes.start_time = start_time;
    new_read->read_common.run_id = run_info_data->protocol_run_id;
    new_read->start_sample = read_data.start_sample;
    new_read->end_sample = read_data.start_sample + read_data.num_samples;
    new_read->read_common.flowcell_id = run_info_data->flow_cell_id;
    new_read->read_common.sequencing_kit = run_info_data->sequencing_kit;
    new_read->read_common.flow_cell_product_code = run_info_data->flow_cell_product_code;
    new_read->read_common.position_id = run_info_data->sequencer_position;
    new_read->read_common.experiment_id = run_info_data->experiment_name;
    new_read->read_common.is_duplex = false;

    // Get the condition_info from the run_info_data to determine if the sequencing kit
    // used has a rapid adapter and which one.
    const auto condition_info = models::ConditionInfo(get_chemistry_key(run_info_data));
    new_read->read_common.rapid_chemistry = condition_info.rapid_chemistry();
    new_read->read_common.chemistry = condition_info.chemistry();

    pod5_end_reason_t end_reason_value{POD5_END_REASON_UNKNOWN};
    char end_reason_string_value[200];
    size_t end_reason_string_value_size = sizeof(end_reason_string_value);

    pod5_error_t pod5_ret =
            pod5_get_end_reason(batch, read_data.end_reason, &end_reason_value,
                                end_reason_string_value, &end_reason_string_value_size);
    if (pod5_ret != POD5_OK) {
        spdlog::error("Failed to get read end_reason {} {}", row, pod5_get_error_string());
    } else if (end_reason_value == POD5_END_REASON_UNBLOCK_MUX_CHANGE ||
               end_reason_value == POD5_END_REASON_MUX_CHANGE) {
        new_read->read_common.attributes.is_end_reason_mux_change = true;
    }

    // Determine the time sorted predecessor of the read
    // if that information is available (primarily used for offline
    // duplex runs).
    if (reads_by_channel.find(read_data.channel) != reads_by_channel.end()) {
        auto& read_id = new_read->read_common.read_id;
        const auto& v = reads_by_channel.at(read_data.channel);
        auto read_id_iter = v.begin() + read_id_to_index.at(read_id);

        if (read_id_iter != v.begin()) {
            new_read->prev_read = std::prev(read_id_iter)->read_id;
        }
        if (std::next(read_id_iter) != v.end()) {
            new_read->next_read = std::next(read_id_iter)->read_id;
        }
    }

    if (pod5_free_run_info(run_info_data) != POD5_OK) {
        spdlog::error("Failed to free run info");
    }
    return new_read;
}

bool can_process_pod5_row(Pod5ReadRecordBatch_t* batch,
                          int row,
                          const std::optional<std::unordered_set<std::string>>& allowed_read_ids,
                          const std::unordered_set<std::string>& ignored_read_ids) {
    uint16_t read_table_version = 0;
    ReadBatchRowInfo_t read_data;
    if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                          &read_table_version) != POD5_OK) {
        spdlog::error("Failed to get read {}", row);
        return false;
    }

    char read_id_tmp[POD5_READ_ID_LEN];
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id");
    }

    std::string read_id_str(read_id_tmp);
    bool read_in_ignore_list = ignored_read_ids.find(read_id_str) != ignored_read_ids.end();
    bool read_in_read_list =
            !allowed_read_ids || (allowed_read_ids->find(read_id_str) != allowed_read_ids->end());
    if (!read_in_ignore_list && read_in_read_list) {
        return true;
    }
    return false;
}

}  // namespace

void DataLoader::load_reads_by_channel(const std::vector<std::filesystem::directory_entry>& files) {
    // If traversal in channel order is required, the following algorithm
    // is used -
    // 1. iterate through all the read metadata to collect channel information
    // across all pod5 files
    // 2. store the read list sorted by channel number
    spdlog::info("> Reading read channel info");
    load_read_channels(files);
    spdlog::info("> Processed read channel info");
    // 3. for each channel, iterate through all files and in each iteration
    // only load the reads that correspond to that channel.
    for (int channel = 0; channel <= m_max_channel; channel++) {
        if (m_reads_by_channel.find(channel) != m_reads_by_channel.end()) {
            // Sort the read ids within a channel by its mux
            // and start time.
            spdlog::trace("Sort channel {}", channel);
            auto& reads = m_reads_by_channel.at(channel);
            std::sort(reads.begin(), reads.end(), [](ReadSortInfo& a, ReadSortInfo& b) {
                if (a.mux != b.mux) {
                    return a.mux < b.mux;
                } else {
                    return a.read_number < b.read_number;
                }
            });
            // Once sorted, create a hash table from read id
            // to index in the sorted list to quickly fetch the
            // read location and its neighbors.
            for (size_t i = 0; i < reads.size(); i++) {
                m_read_id_to_index[reads[i].read_id] = i;
            }
            spdlog::trace("Sorted channel {}", channel);
        }
        for (const auto& entry : files) {
            if (m_loaded_read_count == m_max_reads) {
                break;
            }
            auto entry_path = std::filesystem::path(entry);
            std::string ext = entry_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".fast5") {
                throw std::runtime_error(
                        "Traversing reads by channel is only available for POD5. "
                        "Encountered FAST5 at " +
                        entry_path.string());
            } else if (ext == ".pod5") {
                auto& channel_to_read_ids = m_file_channel_read_order_map.at(entry_path.string());
                auto& read_ids = channel_to_read_ids[channel];
                if (!read_ids.empty()) {
                    load_pod5_reads_from_file_by_read_ids(entry_path.string(), read_ids);
                }
            }
        }
        // Erase sorted list as it's not needed anymore.
        m_reads_by_channel.erase(channel);
    }
}

void DataLoader::load_reads_unrestricted(
        const std::vector<std::filesystem::directory_entry>& files) {
    for (const auto& entry : files) {
        if (m_loaded_read_count == m_max_reads) {
            break;
        }
        spdlog::debug("Load reads from file {}", entry.path().string());
        std::string ext = std::filesystem::path(entry).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".fast5") {
            load_fast5_reads_from_file(entry.path().string());
        } else if (ext == ".pod5") {
            load_pod5_reads_from_file(entry.path().string());
        }
    }
}

void DataLoader::load_reads(const InputFiles& input_files, ReadOrder traversal_order) {
    switch (traversal_order) {
    case ReadOrder::BY_CHANNEL:
        load_reads_by_channel(input_files.get());
        break;
    case ReadOrder::UNRESTRICTED:
        load_reads_unrestricted(input_files.get());
        break;
    default:
        throw std::runtime_error("Unsupported traversal order detected: " +
                                 dorado::to_string(traversal_order));
    }
}

void DataLoader::load_read_channels(const std::vector<std::filesystem::directory_entry>& files) {
    for (const auto& entry : files) {
        auto file_path = std::filesystem::path(entry);
        std::string ext = file_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext != ".pod5") {
            continue;
        }
        pod5_init();

        // Use a std::map to store by sorted channel order.
        m_file_channel_read_order_map.emplace(file_path.string(), channel_to_read_id_t());
        auto& channel_to_read_id = m_file_channel_read_order_map[file_path.string()];

        // Open the file ready for walking:
        Pod5FileReader_t* file = pod5_open_file(file_path.string().c_str());

        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_path.string().c_str(),
                          pod5_get_error_string());
            continue;
        }
        std::size_t batch_count = 0;
        if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
            spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
        }

        for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            Pod5ReadRecordBatch_t* batch = nullptr;
            if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
                spdlog::error("Failed to get batch: {}", pod5_get_error_string());
                continue;
            }

            std::size_t batch_row_count = 0;
            if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
                spdlog::error("Failed to get batch row count");
                continue;
            }

            for (std::size_t row = 0; row < batch_row_count; ++row) {
                uint16_t read_table_version = 0;
                ReadBatchRowInfo_t read_data;
                if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION,
                                                      &read_data, &read_table_version) != POD5_OK) {
                    spdlog::error("Failed to get read {}", row);
                    continue;
                }

                int channel = read_data.channel;

                // Update maximum number of channels encountered.
                m_max_channel = std::max(m_max_channel, channel);

                // Store the read_id in the channel's list.
                ReadID read_id;
                std::memcpy(read_id.data(), read_data.read_id, POD5_READ_ID_SIZE);
                channel_to_read_id[channel].push_back(read_id);

                char read_id_tmp[POD5_READ_ID_LEN];
                if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
                    spdlog::error("Failed to format read id");
                }
                std::string rid(read_id_tmp);
                m_reads_by_channel[channel].push_back({rid, read_data.well, read_data.read_number});
            }

            if (pod5_free_read_batch(batch) != POD5_OK) {
                spdlog::error("Failed to release batch");
            }
        }
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader");
        }
    }
}

void DataLoader::load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                                       const std::vector<ReadID>& read_ids) {
    pod5_init();

    // Open the file ready for walking:
    // TODO: The earlier implementation was caching the pod5 readers into a
    // map and re-using it during each iteration. However, we found a leak
    // in the pod5 traversal API which persists unless the reader is opened
    // and closed everytime. So the caching logic was reverted until the
    // leak is fixed in pod5 API.
    Pod5FileReader_t* file = pod5_open_file(path.c_str());

    if (!file) {
        spdlog::error("Failed to open file {}: {}", path, pod5_get_error_string());
        return;
    }

    auto free_pod5 = [&]() {
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader for file {}", path.c_str());
        }
    };

    auto post = utils::PostCondition(free_pod5);

    std::vector<uint8_t> read_id_array(POD5_READ_ID_SIZE * read_ids.size());
    for (size_t i = 0; i < read_ids.size(); i++) {
        std::memcpy(read_id_array.data() + POD5_READ_ID_SIZE * i, read_ids[i].data(),
                    POD5_READ_ID_SIZE);
    }

    std::size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
    }

    std::vector<std::uint32_t> traversal_batch_counts(batch_count);
    std::vector<std::uint32_t> traversal_batch_rows(read_ids.size());
    size_t find_success_count;
    pod5_error_t err = pod5_plan_traversal(file, read_id_array.data(), read_ids.size(),
                                           traversal_batch_counts.data(),
                                           traversal_batch_rows.data(), &find_success_count);
    if (err != POD5_OK) {
        spdlog::error("Couldn't create plan for {} with reads {}", path, read_ids.size());
        return;
    }

    if (find_success_count != read_ids.size()) {
        spdlog::error("Reads found by plan {}, reads in input {}", find_success_count,
                      read_ids.size());
        throw std::runtime_error("Plan traveral didn't yield correct number of reads");
    }

    // Create static threadpool so it is reused across calls to this function.
    static cxxpool::thread_pool pool{m_num_worker_threads};

    uint32_t row_offset = 0;
    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        if (m_loaded_read_count == m_max_reads) {
            break;
        }
        Pod5ReadRecordBatch_t* batch = nullptr;
        if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
            spdlog::error("Failed to get batch: {}", pod5_get_error_string());
            continue;
        }

        std::vector<std::future<SimplexReadPtr>> futures;
        for (std::size_t row_idx = 0; row_idx < traversal_batch_counts[batch_index]; row_idx++) {
            uint32_t row = traversal_batch_rows[row_idx + row_offset];

            if (can_process_pod5_row(batch, row, m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(pool.push(process_pod5_thread_fn, row, batch, file,
                                            std::cref(path), std::cref(m_reads_by_channel),
                                            std::cref(m_read_id_to_index)));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            initialise_read(read->read_common);
            check_read(read);
            m_pipeline.push_message(std::move(read));
            m_loaded_read_count++;
        }

        if (pod5_free_read_batch(batch) != POD5_OK) {
            spdlog::error("Failed to release batch");
        }

        row_offset += traversal_batch_counts[batch_index];
    }
}

void DataLoader::load_pod5_reads_from_file(const std::string& path) {
    pod5_init();

    // Open the file ready for walking:
    Pod5FileReader_t* file = pod5_open_file(path.c_str());

    if (!file) {
        spdlog::error("Failed to open file {}: {}", path, pod5_get_error_string());
    }

    std::size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
    }

    cxxpool::thread_pool pool{m_num_worker_threads};

    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        if (m_loaded_read_count == m_max_reads) {
            break;
        }
        Pod5ReadRecordBatch_t* batch = nullptr;
        if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
            spdlog::error("Failed to get batch: {}", pod5_get_error_string());
        }

        std::size_t batch_row_count = 0;
        if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
            spdlog::error("Failed to get batch row count");
        }
        batch_row_count = std::min(batch_row_count, m_max_reads - m_loaded_read_count);

        std::vector<std::future<SimplexReadPtr>> futures;

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            // TODO - check the read ID here, for each one, only send the row if it is in the list of ones we care about

            if (can_process_pod5_row(batch, int(row), m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(pool.push(process_pod5_thread_fn, row, batch, file,
                                            std::cref(path), std::cref(m_reads_by_channel),
                                            std::cref(m_read_id_to_index)));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            initialise_read(read->read_common);
            check_read(read);
            m_pipeline.push_message(std::move(read));
            m_loaded_read_count++;
        }

        if (pod5_free_read_batch(batch) != POD5_OK) {
            spdlog::error("Failed to release batch");
        }
    }
    if (pod5_close_and_free_reader(file) != POD5_OK) {
        spdlog::error("Failed to close and free POD5 reader");
    }
}

void DataLoader::load_fast5_reads_from_file(const std::string& path) {
    // Read the file into a vector of torch tensors
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    HighFive::Group reads = file.getGroup("/");
    int num_reads = int(reads.getNumberObjects());

    for (int i = 0; i < num_reads && m_loaded_read_count < m_max_reads; i++) {
        auto read_id = reads.getObjectName(i);
        HighFive::Group read = reads.getGroup(read_id);

        // Fetch the digitisation parameters
        HighFive::Group channel_id_group = read.getGroup("channel_id");
        HighFive::Attribute digitisation_attr = channel_id_group.getAttribute("digitisation");
        HighFive::Attribute range_attr = channel_id_group.getAttribute("range");
        HighFive::Attribute offset_attr = channel_id_group.getAttribute("offset");
        HighFive::Attribute sampling_rate_attr = channel_id_group.getAttribute("sampling_rate");
        HighFive::Attribute channel_number_attr = channel_id_group.getAttribute("channel_number");

        int32_t channel_number;
        if (channel_number_attr.getDataType().string().substr(0, 6) == "String") {
            std::string channel_number_string;
            string_reader(channel_number_attr, channel_number_string);
            std::istringstream channel_stream(channel_number_string);
            channel_stream >> channel_number;
        } else {
            channel_number_attr.read(channel_number);
        }

        float digitisation;
        digitisation_attr.read(digitisation);
        float range;
        range_attr.read(range);
        float offset;
        offset_attr.read(offset);
        float sampling_rate;
        sampling_rate_attr.read(sampling_rate);

        HighFive::Group raw = read.getGroup("Raw");
        auto ds = raw.getDataSet("Signal");
        if (ds.getDataType().string() != "Integer16") {
            throw std::runtime_error("Invalid FAST5 Signal data type of " +
                                     ds.getDataType().string());
        }

        auto options = at::TensorOptions().dtype(at::kShort);
        auto samples = at::empty(ds.getElementCount(), options);
        ds.read(samples.data_ptr<int16_t>());

        HighFive::Attribute mux_attr = raw.getAttribute("start_mux");
        HighFive::Attribute read_number_attr = raw.getAttribute("read_number");
        HighFive::Attribute start_time_attr = raw.getAttribute("start_time");
        HighFive::Attribute read_id_attr = raw.getAttribute("read_id");
        uint32_t mux;
        uint32_t read_number;
        uint64_t start_time;
        mux_attr.read(mux);
        read_number_attr.read(read_number);
        start_time_attr.read(start_time);
        string_reader(read_id_attr, read_id);

        std::string filename = std::filesystem::path(path).filename().string();

        HighFive::Group tracking_id_group = read.getGroup("tracking_id");
        std::string exp_start_time = get_string_attribute(tracking_id_group, "exp_start_time");
        std::string flow_cell_id = get_string_attribute(tracking_id_group, "flow_cell_id");
        std::string flow_cell_product_code =
                get_string_attribute(tracking_id_group, "flow_cell_product_code");
        std::string device_id = get_string_attribute(tracking_id_group, "device_id");
        std::string group_protocol_id =
                get_string_attribute(tracking_id_group, "group_protocol_id");

        auto start_time_str = utils::adjust_time(exp_start_time,
                                                 static_cast<uint32_t>(start_time / sampling_rate));

        auto new_read = std::make_unique<SimplexRead>();
        new_read->read_common.sample_rate = uint64_t(sampling_rate);
        new_read->read_common.raw_data = samples;
        new_read->digitisation = digitisation;
        new_read->range = range;
        new_read->offset = offset;
        new_read->scaling = range / digitisation;
        new_read->read_common.read_id = read_id;
        new_read->read_common.num_trimmed_samples = 0;
        new_read->read_common.attributes.mux = mux;
        new_read->read_common.attributes.read_number = read_number;
        new_read->read_common.attributes.channel_number = channel_number;
        new_read->read_common.attributes.start_time = start_time_str;
        new_read->read_common.attributes.filename = filename;
        new_read->read_common.flowcell_id = flow_cell_id;
        new_read->read_common.flow_cell_product_code = flow_cell_product_code;
        new_read->read_common.position_id = device_id;
        new_read->read_common.experiment_id = group_protocol_id;
        new_read->read_common.is_duplex = false;

        if (!m_allowed_read_ids || (m_allowed_read_ids->find(new_read->read_common.read_id) !=
                                    m_allowed_read_ids->end())) {
            initialise_read(new_read->read_common);
            m_pipeline.push_message(std::move(new_read));
            m_loaded_read_count++;
        }
    }
}

void DataLoader::initialise_read(ReadCommon& read_common) const {
    for (const auto& initialiser : m_read_initialisers) {
        initialiser(read_common);
    }
}

void DataLoader::check_read(const SimplexReadPtr& read) {
    if (read->read_common.chemistry == models::Chemistry::UNKNOWN &&
        m_log_unknown_chemistry.exchange(false)) {
        spdlog::warn(
                "Could not determine sequencing Chemistry from read data - "
                "some features might be disabled");
    }
}

DataLoader::DataLoader(Pipeline& pipeline,
                       const std::string& device,
                       size_t num_worker_threads,
                       size_t max_reads,
                       std::optional<std::unordered_set<std::string>> read_list,
                       std::unordered_set<std::string> read_ignore_list)
        : m_pipeline(pipeline),
          m_device(device),
          m_num_worker_threads(num_worker_threads),
          m_allowed_read_ids(std::move(read_list)),
          m_ignored_read_ids(std::move(read_ignore_list)) {
    m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
    assert(m_num_worker_threads > 0);
    static std::once_flag vbz_init_flag;
    std::call_once(vbz_init_flag, vbz_register);
}

std::optional<DataLoader::InputFiles> DataLoader::InputFiles::search(
        const std::filesystem::path& path,
        bool recursive) {
    auto entries = filter_fast5_for_mixed_datasets(utils::fetch_directory_entries(path, recursive));
    if (!entries.has_value()) {
        return std::nullopt;
    }

    // Intentionally returning a valid object even if there are 0 entries since duplex uses that
    // to differentiate between different modes of operation.
    InputFiles files;
    files.m_entries = std::move(*entries);
    return files;
}

const std::vector<std::filesystem::directory_entry>& DataLoader::InputFiles::get() const {
    return m_entries;
}

stats::NamedStats DataLoader::sample_stats() const {
    return stats::NamedStats{{"loaded_read_count", static_cast<double>(m_loaded_read_count)}};
}
}  // namespace dorado
