#include "DataLoader.h"

#include "models/kits.h"
#include "read_pipeline/ReadPipeline.h"
#include "read_pipeline/messages.h"
#include "utils/PostCondition.h"
#include "utils/fs_utils.h"
#include "utils/thread_naming.h"
#include "utils/time_utils.h"
#include "utils/types.h"

#include <ATen/Functions.h>
#include <pod5_format/c_api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <vector>

namespace dorado {

namespace {

// ReadID should be a drop-in replacement for read_id_t
static_assert(sizeof(dorado::ReadID) == sizeof(read_id_t));

// 37 = number of bytes in UUID (32 hex digits + 4 dashes + null terminator)
const uint32_t POD5_READ_ID_LEN = 37;

std::vector<std::filesystem::directory_entry> collect_pod5_dataset(
        const std::vector<std::filesystem::directory_entry>& files) {
    std::vector<std::filesystem::directory_entry> pod5_entries;
    bool fast5_found = false;

    for (const auto& entry : files) {
        const auto ext = utils::get_extension(entry);
        if (ext == ".fast5") {
            fast5_found = true;
        } else if (ext == ".pod5") {
            pod5_entries.push_back(entry);
        }
    }

    if (fast5_found && pod5_entries.empty()) {
        spdlog::error(
                "FAST5 support in Dorado was removed in version 1.0.0. "
                "Please convert your dataset to POD5: "
                "https://pod5-file-format.readthedocs.io/en/latest/docs/"
                "tools.html#pod5-convert-fast5");
        throw std::runtime_error("FAST5 files are not supported.");
    }

    if (fast5_found && !pod5_entries.empty()) {
        spdlog::warn(
                "Skipping FAST5 files as support was dropped in Dorado version 1.0.0. "
                "Please convert your FAST5 dataset to POD5: "
                "https://pod5-file-format.readthedocs.io/en/latest/docs/"
                "tools.html#pod5-convert-fast5");
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
        const Pod5ReadRecordBatch* batch,
        const Pod5FileReader* file,
        const std::string& path,
        const std::unordered_map<int, std::vector<DataLoader::ReadSortInfo>>& reads_by_channel,
        const std::unordered_map<std::string, size_t>& read_id_to_index) {
    utils::set_thread_name("process_pod5");
    uint16_t read_table_version = 0;
    ReadBatchRowInfo_t read_data{};
    if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                          &read_table_version) != POD5_OK) {
        spdlog::error("Failed to get read {}: {}", row, pod5_get_error_string());
        return nullptr;
    }

    //Retrieve global information for the run
    RunInfoDictData_t* run_info_data = nullptr;
    if (pod5_get_run_info(batch, read_data.run_info, &run_info_data) != POD5_OK) {
        spdlog::error("Failed to get Run Info {}: {}", row, pod5_get_error_string());
        return nullptr;
    }
    auto cleanup = utils::PostCondition([&run_info_data] {
        if (pod5_free_run_info(run_info_data) != POD5_OK) {
            spdlog::error("Failed to free run info: {}", pod5_get_error_string());
        }
    });

    auto run_acquisition_start_time_ms = run_info_data->acquisition_start_time_ms;
    auto run_sample_rate = run_info_data->sample_rate;

    char read_id_tmp[POD5_READ_ID_LEN]{};
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id: {}", pod5_get_error_string());
        return nullptr;
    }
    std::string read_id_str(read_id_tmp);

    auto options = at::TensorOptions().dtype(at::kShort);
    auto samples = at::empty(read_data.num_samples, options);

    if (pod5_get_read_complete_signal(file, batch, row, read_data.num_samples,
                                      samples.data_ptr<int16_t>()) != POD5_OK) {
        spdlog::error("Failed to get read {} signal: {}", row, pod5_get_error_string());
        return nullptr;
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
    char end_reason_string_value[200]{};
    size_t end_reason_string_value_size = sizeof(end_reason_string_value);

    pod5_error_t pod5_ret =
            pod5_get_end_reason(batch, read_data.end_reason, &end_reason_value,
                                end_reason_string_value, &end_reason_string_value_size);
    if (pod5_ret != POD5_OK) {
        spdlog::error("Failed to get read end_reason {}: {}", row, pod5_get_error_string());
        return nullptr;
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

    return new_read;
}

bool can_process_pod5_row(const Pod5ReadRecordBatch_t* batch,
                          int row,
                          const std::optional<std::unordered_set<std::string>>& allowed_read_ids,
                          const std::unordered_set<std::string>& ignored_read_ids) {
    uint16_t read_table_version = 0;
    ReadBatchRowInfo_t read_data{};
    if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data,
                                          &read_table_version) != POD5_OK) {
        spdlog::error("Failed to get read {}: {}", row, pod5_get_error_string());
        return false;
    }

    char read_id_tmp[POD5_READ_ID_LEN]{};
    if (pod5_format_read_id(read_data.read_id, read_id_tmp) != POD5_OK) {
        spdlog::error("Failed to format read id: {}", pod5_get_error_string());
        return false;
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

            if (!utils::has_pod5_extension(entry)) {
                throw std::logic_error("Expected pod5 file");
            }

            const auto path = std::filesystem::path(entry);
            auto& channel_to_read_ids = m_file_channel_read_order_map.at(path.string());
            auto& read_ids = channel_to_read_ids[channel];
            if (!read_ids.empty()) {
                load_pod5_reads_from_file_by_read_ids(path.string(), read_ids);
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
        if (ext == ".pod5") {
            load_pod5_reads_from_file(entry.path().string());
        }
    }
}

void DataLoader::load_reads(const InputFiles& input_files, ReadOrder traversal_order) {
    if (pod5_init() != POD5_OK) {
        throw std::runtime_error(
                fmt::format("Failed to initialise POD5: {}", pod5_get_error_string()));
    }
    auto pod5_cleanup = utils::PostCondition([] { pod5_terminate(); });

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

        // Use a std::map to store by sorted channel order.
        const auto file_string = file_path.string();
        auto& channel_to_read_id = m_file_channel_read_order_map[file_string];

        // Open the file ready for walking:
        Pod5FileReader_t* file = pod5_open_file(file_string.c_str());
        if (!file) {
            spdlog::error("Failed to open file {}: {}", file_string, pod5_get_error_string());
            continue;
        }
        auto cleanup_file = utils::PostCondition([&file, &file_string] {
            if (pod5_close_and_free_reader(file) != POD5_OK) {
                spdlog::error("Failed to close and free POD5 reader for file {}: {}", file_string,
                              pod5_get_error_string());
            }
        });

        std::size_t batch_count = 0;
        if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
            spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
            continue;
        }

        for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
            Pod5ReadRecordBatch_t* batch = nullptr;
            if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
                spdlog::error("Failed to get batch: {}", pod5_get_error_string());
                continue;
            }
            auto cleanup_batch = utils::PostCondition([&batch] {
                if (pod5_free_read_batch(batch) != POD5_OK) {
                    spdlog::error("Failed to release batch: {}", pod5_get_error_string());
                }
            });

            std::size_t batch_row_count = 0;
            if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
                spdlog::error("Failed to get batch row count: {}", pod5_get_error_string());
                continue;
            }

            for (std::size_t row = 0; row < batch_row_count; ++row) {
                uint16_t read_table_version = 0;
                ReadBatchRowInfo_t read_data{};
                if (pod5_get_read_batch_row_info_data(batch, row, READ_BATCH_ROW_INFO_VERSION,
                                                      &read_data, &read_table_version) != POD5_OK) {
                    spdlog::error("Failed to get read {}: ", row, pod5_get_error_string());
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
                    spdlog::error("Failed to format read id: {}", pod5_get_error_string());
                }
                std::string rid(read_id_tmp);
                m_reads_by_channel[channel].push_back({rid, read_data.well, read_data.read_number});
            }
        }
    }
}

void DataLoader::load_pod5_reads_from_file_by_read_ids(const std::string& path,
                                                       const std::vector<ReadID>& read_ids) {
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
    auto cleanup_file = utils::PostCondition([&file, &path] {
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader for file {}: {}", path,
                          pod5_get_error_string());
        }
    });

    std::vector<uint8_t> read_id_array(POD5_READ_ID_SIZE * read_ids.size());
    for (size_t i = 0; i < read_ids.size(); i++) {
        std::memcpy(read_id_array.data() + POD5_READ_ID_SIZE * i, read_ids[i].data(),
                    POD5_READ_ID_SIZE);
    }

    std::size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
        return;
    }

    std::vector<std::uint32_t> traversal_batch_counts(batch_count);
    std::vector<std::uint32_t> traversal_batch_rows(read_ids.size());
    size_t find_success_count;
    pod5_error_t err = pod5_plan_traversal(file, read_id_array.data(), read_ids.size(),
                                           traversal_batch_counts.data(),
                                           traversal_batch_rows.data(), &find_success_count);
    if (err != POD5_OK) {
        spdlog::error("Couldn't create plan for {} with reads {}: {}", path, read_ids.size(),
                      pod5_get_error_string());
        return;
    }

    if (find_success_count != read_ids.size()) {
        spdlog::error("Reads found by plan {}, reads in input {}", find_success_count,
                      read_ids.size());
        throw std::runtime_error("Plan traveral didn't yield correct number of reads");
    }

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
        auto cleanup_batch = utils::PostCondition([&batch] {
            if (pod5_free_read_batch(batch) != POD5_OK) {
                spdlog::error("Failed to release batch: {}", pod5_get_error_string());
            }
        });

        std::vector<std::future<SimplexReadPtr>> futures;
        for (std::size_t row_idx = 0; row_idx < traversal_batch_counts[batch_index]; row_idx++) {
            uint32_t row = traversal_batch_rows[row_idx + row_offset];

            if (can_process_pod5_row(batch, row, m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(m_thread_pool.push([row, batch, file, &path, this] {
                    return process_pod5_thread_fn(row, batch, file, path, m_reads_by_channel,
                                                  m_read_id_to_index);
                }));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            if (!read) {
                continue;
            }
            initialise_read(read->read_common);
            check_read(read);
            m_pipeline.push_message(std::move(read));
            m_loaded_read_count++;
        }

        row_offset += traversal_batch_counts[batch_index];
    }
}

void DataLoader::load_pod5_reads_from_file(const std::string& path) {
    // Open the file ready for walking:
    Pod5FileReader_t* file = pod5_open_file(path.c_str());
    if (!file) {
        spdlog::error("Failed to open file {}: {}", path, pod5_get_error_string());
        return;
    }
    auto file_cleanup = utils::PostCondition([&file, &path] {
        if (pod5_close_and_free_reader(file) != POD5_OK) {
            spdlog::error("Failed to close and free POD5 reader for file {}: {}", path,
                          pod5_get_error_string());
        }
    });

    std::size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        spdlog::error("Failed to query batch count: {}", pod5_get_error_string());
        return;
    }

    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        if (m_loaded_read_count == m_max_reads) {
            break;
        }
        Pod5ReadRecordBatch_t* batch = nullptr;
        if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
            spdlog::error("Failed to get batch: {}", pod5_get_error_string());
            continue;
        }
        auto cleanup_batch = utils::PostCondition([&batch] {
            if (pod5_free_read_batch(batch) != POD5_OK) {
                spdlog::error("Failed to release batch: {}", pod5_get_error_string());
            }
        });

        std::size_t batch_row_count = 0;
        if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
            spdlog::error("Failed to get batch row count: {}", pod5_get_error_string());
            continue;
        }
        batch_row_count = std::min(batch_row_count, m_max_reads - m_loaded_read_count);

        std::vector<std::future<SimplexReadPtr>> futures;

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            // TODO - check the read ID here, for each one, only send the row if it is in the list of ones we care about

            if (can_process_pod5_row(batch, int(row), m_allowed_read_ids, m_ignored_read_ids)) {
                futures.push_back(m_thread_pool.push([row, batch, file, &path, this] {
                    return process_pod5_thread_fn(row, batch, file, path, m_reads_by_channel,
                                                  m_read_id_to_index);
                }));
            }
        }

        for (auto& v : futures) {
            auto read = v.get();
            initialise_read(read->read_common);
            check_read(read);
            m_pipeline.push_message(std::move(read));
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
          m_thread_pool(num_worker_threads),
          m_allowed_read_ids(std::move(read_list)),
          m_ignored_read_ids(std::move(read_ignore_list)) {
    m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
    assert(m_thread_pool.n_threads() > 0);
}

DataLoader::InputFiles DataLoader::InputFiles::search_pod5s(const std::filesystem::path& path,
                                                            bool recursive) {
    auto entries = collect_pod5_dataset(utils::fetch_directory_entries(path, recursive));

    // Intentionally returning a valid object even if there are 0 entries since duplex uses that
    // to differentiate between different modes of operation.
    InputFiles files;
    files.m_entries = std::move(entries);
    return files;
}

const std::vector<std::filesystem::directory_entry>& DataLoader::InputFiles::get() const {
    return m_entries;
}

}  // namespace dorado
