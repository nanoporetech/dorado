#include "DataLoader.h"

#include "../read_pipeline/ReadPipeline.h"
#include "pod5_format/c_api.h"
#include "utils/compat_utils.h"
#include "vbz_plugin_user_utils.h"

#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>

#include <cctype>
#include <ctime>
#include <filesystem>

namespace {
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
};

std::string adjust_time(const std::string& time_stamp, uint32_t offset) {
    // Expects the time to be encoded like "2017-09-12T9:50:12Z".
    // Adds the offset (in seconds) to the timeStamp.
    std::tm base_time = {};
    strptime(time_stamp.c_str(), "%Y-%m-%dT%H:%M:%SZ", &base_time);
    time_t timeObj = mktime(&base_time);
    timeObj += offset;
    std::tm* new_time = gmtime(&timeObj);
    char buff[32];
    strftime(buff, 32, "%FT%TZ", new_time);
    return std::string(buff);
}
} /* anonymous namespace */

void DataLoader::load_reads(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "Requested input path " << path << " does not exist!" << std::endl;
        m_read_sink.terminate();
        return;
    }
    if (!std::filesystem::is_directory(path)) {
        std::cerr << "Requested input path " << path << " is not a directory!" << std::endl;
        m_read_sink.terminate();
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        std::string ext = std::filesystem::path(entry).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (ext == ".fast5") {
            load_fast5_reads_from_file(entry.path().string());
        } else if (ext == ".pod5") {
            load_pod5_reads_from_file(entry.path().string());
        }
    }
    m_read_sink.terminate();
    std::cerr << "> Loaded " << m_loaded_read_count << " reads" << std::endl;
}

void DataLoader::load_pod5_reads_from_file(const std::string& path) {
    pod5_init();

    // Open the file ready for walking:
    Pod5FileReader_t* file = pod5_open_combined_file(path.c_str());

    if (!file) {
        std::cerr << "Failed to open file " << path.c_str() << ": " << pod5_get_error_string()
                  << "\n";
    }

    std::size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        std::cerr << "Failed to query batch count: " << pod5_get_error_string() << "\n";
    }

    size_t read_count = 0;
    std::size_t samples_read = 0;

    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
        Pod5ReadRecordBatch_t* batch = nullptr;
        if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
            std::cerr << "Failed to get batch: " << pod5_get_error_string() << "\n";
        }

        std::size_t batch_row_count = 0;
        if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
            std::cerr << "Failed to get batch row count\n";
        }

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            uint8_t read_id[16];
            int16_t pore = 0;
            int16_t calibration_idx = 0;
            uint32_t read_number = 0;
            uint64_t start_sample = 0;
            float median_before = 0.0f;
            int16_t end_reason = 0;
            int16_t run_info = 0;
            int64_t signal_row_count = 0;
            if (pod5_get_read_batch_row_info(
                        batch, row, read_id, &pore, &calibration_idx, &read_number, &start_sample,
                        &median_before, &end_reason, &run_info, &signal_row_count) != POD5_OK) {
                std::cerr << "Failed to get read " << row << "\n";
            }
            read_count += 1;

            char read_id_tmp[37];
            pod5_error_t err = pod5_format_read_id(read_id, read_id_tmp);
            std::string read_id_str(read_id_tmp);

            // Now read out the calibration params:
            CalibrationDictData_t* calib_data = nullptr;
            if (pod5_get_calibration(batch, calibration_idx, &calib_data) != POD5_OK) {
                std::cerr << "Failed to get read " << row
                          << " calibration_idx data: " << pod5_get_error_string() << "\n";
            }

            // Find the absolute indices of the signal rows in the signal table
            std::vector<std::uint64_t> signal_rows_indices(signal_row_count);
            if (pod5_get_signal_row_indices(batch, row, signal_row_count,
                                            signal_rows_indices.data()) != POD5_OK) {
                std::cerr << "Failed to get read " << row
                          << " signal row indices: " << pod5_get_error_string() << "\n";
            }

            // Find the locations of each row in signal batches:
            std::vector<SignalRowInfo_t*> signal_rows(signal_row_count);
            if (pod5_get_signal_row_info(file, signal_row_count, signal_rows_indices.data(),
                                         signal_rows.data()) != POD5_OK) {
                std::cerr << "Failed to get read " << row
                          << " signal row locations: " << pod5_get_error_string() << "\n";
            }

            std::size_t total_sample_count = 0;
            for (std::size_t i = 0; i < signal_row_count; ++i) {
                total_sample_count += signal_rows[i]->stored_sample_count;
            }

            std::vector<std::int16_t> samples(total_sample_count);
            std::size_t samples_read_so_far = 0;
            for (std::size_t i = 0; i < signal_row_count; ++i) {
                if (pod5_get_signal(file, signal_rows[i], signal_rows[i]->stored_sample_count,
                                    samples.data() + samples_read_so_far) != POD5_OK) {
                    std::cerr << "Failed to get read " << row
                              << " signal: " << pod5_get_error_string() << "\n";
                }

                samples_read_so_far += signal_rows[i]->stored_sample_count;
            }

            std::vector<float> floatTmp(samples.begin(), samples.end());

            samples_read += samples.size();

            auto new_read = std::make_shared<Read>();

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            new_read->raw_data = torch::from_blob(floatTmp.data(), floatTmp.size(), options)
                                         .clone()
                                         .to(m_device);
            float scale = calib_data->scale;
            float offset = calib_data->offset;

            new_read->scale = calib_data->scale;
            new_read->scale_set = true;
            new_read->offset = calib_data->offset;
            new_read->read_id = read_id_str;

            m_read_sink.push_read(new_read);
            m_loaded_read_count++;

            pod5_release_calibration(calib_data);
            pod5_free_signal_row_info(signal_row_count, signal_rows.data());
        }

        if (pod5_free_read_batch(batch) != POD5_OK) {
            std::cerr << "Failed to release batch\n";
        }
    }
    pod5_close_and_free_reader(file);
}

void DataLoader::load_fast5_reads_from_file(const std::string& path) {
    // Read the file into a vector of torch tensors
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    HighFive::Group reads = file.getGroup("/");
    int num_reads = reads.getNumberObjects();

    for (int i = 0; i < num_reads; i++) {
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
        if (ds.getDataType().string() != "Integer16")
            throw std::runtime_error("Invalid FAST5 Signal data type of " +
                                     ds.getDataType().string());
        std::vector<int16_t> tmp;
        ds.read(tmp);
        std::vector<float> floatTmp(tmp.begin(), tmp.end());

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

        std::string fast5_filename = std::filesystem::path(path).filename().string();

        HighFive::Group tracking_id_group = read.getGroup("tracking_id");
        HighFive::Attribute exp_start_time_attr = tracking_id_group.getAttribute("exp_start_time");
        std::string exp_start_time;
        string_reader(exp_start_time_attr, exp_start_time);

        auto start_time_str =
                adjust_time(exp_start_time, static_cast<uint32_t>(start_time / sampling_rate));

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto new_read = std::make_shared<Read>();
        new_read->raw_data =
                torch::from_blob(floatTmp.data(), floatTmp.size(), options).clone().to(m_device);
        new_read->digitisation = digitisation;
        new_read->range = range;
        new_read->offset = offset;
        new_read->read_id = read_id;
        new_read->num_samples = floatTmp.size();
        new_read->num_trimmed_samples = floatTmp.size();  // same value until we actually trim
        new_read->attributes.mux = mux;
        new_read->attributes.read_number = read_number;
        new_read->attributes.channel_number = channel_number;
        new_read->attributes.start_time = start_time_str;
        new_read->attributes.fast5_filename = fast5_filename;

        m_read_sink.push_read(new_read);
        m_loaded_read_count++;
    }
}

DataLoader::DataLoader(ReadSink& read_sink, const std::string& device)
        : m_read_sink(read_sink), m_device(device) {
    static std::once_flag vbz_init_flag;
    std::call_once(vbz_init_flag, vbz_register);
}
