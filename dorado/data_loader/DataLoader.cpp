#include "DataLoader.h"

#include "../read_pipeline/ReadPipeline.h"
#include "../utils/compat_utils.h"
#include "cxxpool.h"
#ifdef USE_POD5
#include "pod5_format/c_api.h"
#endif
#ifdef USE_FAST5
#include "vbz_plugin_user_utils.h"
#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#endif

#include <spdlog/spdlog.h>

#include <cctype>
#include <ctime>
#include <filesystem>

#include "slow5/slow5.h"
#include "slow5_extra.h"
#include "slow5_thread.h"

namespace {
#ifdef USE_FAST5
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
#endif

std::string get_string_timestamp_from_unix_time(time_t time_stamp_ms) {
    static std::mutex timestamp_mtx;
    std::unique_lock lock(timestamp_mtx);
    //Convert a time_t (seconds from UNIX epoch) to a timestamp in %Y-%m-%dT%H:%M:%SZ format
    auto time_stamp_s = time_stamp_ms / 1000;
    int num_ms = time_stamp_ms % 1000;
    char buf[32];
    struct tm ts;
    ts = *gmtime(&time_stamp_s);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S.", &ts);
    std::string time_stamp_str = std::string(buf);
    time_stamp_str += std::to_string(num_ms);  // add ms
    time_stamp_str += "+00:00";                //add zero timezone
    return time_stamp_str;
}

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

#ifdef USE_POD5
std::shared_ptr<dorado::Read> process_pod5_read(size_t row,
                                                Pod5ReadRecordBatch* batch,
                                                Pod5FileReader* file,
                                                const std::string path,
                                                std::string device) {
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

    char read_id_tmp[37];
    pod5_error_t err = pod5_format_read_id(read_data.read_id, read_id_tmp);
    std::string read_id_str(read_id_tmp);

    auto options = torch::TensorOptions().dtype(torch::kInt16);
    auto samples = torch::empty(read_data.num_samples, options);

    if (pod5_get_read_complete_signal(file, batch, row, read_data.num_samples,
                                      samples.data_ptr<int16_t>()) != POD5_OK) {
        spdlog::error("Failed to get read {} signal: {}", row, pod5_get_error_string());
    }

    auto new_read = std::make_shared<dorado::Read>();
    new_read->raw_data = samples;
    new_read->sample_rate = run_sample_rate;
    auto start_time_ms =
            run_acquisition_start_time_ms + ((read_data.start_sample * 1000) / run_sample_rate);
    auto start_time = get_string_timestamp_from_unix_time(start_time_ms);
    new_read->scaling = read_data.calibration_scale;
    new_read->offset = read_data.calibration_offset;
    new_read->read_id = std::move(read_id_str);
    new_read->num_trimmed_samples = 0;
    new_read->attributes.read_number = read_data.read_number;
    new_read->attributes.fast5_filename = std::filesystem::path(path.c_str()).filename().string();
    new_read->attributes.mux = read_data.well;
    new_read->attributes.channel_number = read_data.channel;
    new_read->attributes.start_time = start_time;

    return new_read;
}
#endif

} /* anonymous namespace */

namespace dorado {

void DataLoader::load_reads(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Requested input path {} does not exist!", path);
        m_read_sink.terminate();
        return;
    }
    if(!std::filesystem::is_directory(path)) {
        std::string ext = std::filesystem::path(path).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
#ifdef USE_FAST5
        if(ext == ".fast5") {
            load_fast5_reads_from_file(path);
        }
#endif
#ifdef USE_POD5
        if(ext == ".pod5") {
            load_pod5_reads_from_file(path);
        }
#endif
        if(ext == ".slow5" || ext == ".blow5") {
            load_slow5_reads_from_file(path);
        }
    }else{
        for (const auto & entry : std::filesystem::directory_iterator(path)) {
            std::string ext = std::filesystem::path(entry).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
#ifdef USE_FAST5
            if(ext == ".fast5") {
                load_fast5_reads_from_file(entry.path().string());
            }
#endif
#ifdef USE_POD5
            if(ext == ".pod5") {
                load_pod5_reads_from_file(entry.path().string());
            }
#endif
            if(ext == ".slow5" || ext == ".blow5") {
                load_slow5_reads_from_file(entry.path().string());
            }
        }
    }
    m_read_sink.terminate();
}

#ifdef USE_POD5
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

        std::vector<std::future<std::shared_ptr<Read>>> futures;

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            futures.push_back(pool.push(process_pod5_read, row, batch, file, path, m_device));
        }

        for (auto& v : futures) {
            auto read = v.get();
            m_read_sink.push_read(read);
            m_loaded_read_count++;
        }

        if (pod5_free_read_batch(batch) != POD5_OK) {
            spdlog::error("Failed to release batch");
        }
    }
    pod5_close_and_free_reader(file);
}
#endif
#ifdef USE_FAST5
void DataLoader::load_fast5_reads_from_file(const std::string& path) {
    // Read the file into a vector of torch tensors
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    HighFive::Group reads = file.getGroup("/");
    int num_reads = reads.getNumberObjects();

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
        if (ds.getDataType().string() != "Integer16")
            throw std::runtime_error("Invalid FAST5 Signal data type of " +
                                     ds.getDataType().string());

        auto options = torch::TensorOptions().dtype(torch::kInt16);
        auto samples = torch::empty(ds.getElementCount(), options);
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

        std::string fast5_filename = std::filesystem::path(path).filename().string();

        HighFive::Group tracking_id_group = read.getGroup("tracking_id");
        HighFive::Attribute exp_start_time_attr = tracking_id_group.getAttribute("exp_start_time");
        std::string exp_start_time;
        string_reader(exp_start_time_attr, exp_start_time);

        auto start_time_str =
                adjust_time(exp_start_time, static_cast<uint32_t>(start_time / sampling_rate));

        auto new_read = std::make_shared<Read>();
        new_read->sample_rate = sampling_rate;
        new_read->raw_data = samples;
        new_read->digitisation = digitisation;
        new_read->range = range;
        new_read->offset = offset;
        new_read->scaling = range / digitisation;
        new_read->read_id = read_id;
        new_read->num_trimmed_samples = 0;
        new_read->attributes.mux = mux;
        new_read->attributes.read_number = read_number;
        new_read->attributes.channel_number = channel_number;
        new_read->attributes.start_time = start_time_str;
        new_read->attributes.fast5_filename = fast5_filename;

        m_read_sink.push_read(new_read);
        m_loaded_read_count++;
    }
}
#endif

void create_read_data(core_t *core, db_t *db, int32_t i) {
    //
    struct slow5_rec *rec = NULL;
    if (slow5_rec_depress_parse(&db->mem_records[i], &db->mem_bytes[i], NULL, &rec, core->fp) != 0) {
        exit(EXIT_FAILURE);
    } else {
        free(db->mem_records[i]);
    }
    auto new_read = std::make_shared<dorado::Read>();

    //
    std::vector<int16_t> tmp(rec->raw_signal,rec->raw_signal+rec->len_raw_signal);
//    std::vector<float> floatTmp(tmp.begin(), tmp.end());

    int ret = 0;
    uint64_t start_time = slow5_aux_get_uint64(rec, "start_time", &ret);
    if(ret!=0){
        throw std::runtime_error("Error in getting auxiliary attribute 'start_time' from the file.");
    }
    ret = 0;
    uint32_t mux = slow5_aux_get_uint8(rec, "start_mux", &ret);
    if(ret!=0){
        throw std::runtime_error("Error in getting auxiliary attribute 'start_mux' from the file.");
    }
    ret = 0;
    int32_t read_number = slow5_aux_get_int32(rec, "read_number", &ret);
    if(ret!=0){
        throw std::runtime_error("Error in getting auxiliary attribute 'read_number' from the file.");
    }
    ret = 0;
    uint64_t len;
    std::string channel_number_str = slow5_aux_get_string(rec, "channel_number", &len, &ret);
    if(ret!=0){
        throw std::runtime_error("Error in getting auxiliary attribute 'channel_number' from the file.");
    }
    int32_t channel_number = static_cast<int32_t>(std::stol(channel_number_str));
    char* exp_start_time = slow5_hdr_get("exp_start_time", rec->read_group, core->fp->header);
    if(!exp_start_time){
        throw std::runtime_error("exp_start_time is missing");
    }

    auto start_time_str = adjust_time(exp_start_time, static_cast<uint32_t>(start_time / rec->sampling_rate));

//    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto options = torch::TensorOptions().dtype(torch::kInt16);
    new_read->sample_rate = rec->sampling_rate;
    new_read->raw_data = torch::from_blob(tmp.data(), tmp.size(), options).clone().to(core->m_device_);
    new_read->digitisation = rec->digitisation;
    new_read->range = rec->range;
    new_read->scaling = rec->range / rec->digitisation;
    new_read->offset = rec->offset;
    new_read->read_id = rec->read_id;
    new_read->num_trimmed_samples = 0;
    new_read->attributes.mux = mux;
    new_read->attributes.read_number = read_number;
    new_read->attributes.channel_number = channel_number;
    new_read->attributes.start_time = start_time_str;
    new_read->attributes.fast5_filename = core->fp->meta.pathname;
    //
    db->read_data_ptrs[i] = new_read;
    slow5_rec_free(rec);
}

void DataLoader::load_slow5_reads_from_file(const std::string& path){
    slow5_file_t *sp = slow5_open(path.c_str(),"r");
    if(sp==NULL){
        fprintf(stderr,"Error in opening file\n");
        exit(EXIT_FAILURE);
    }
    int64_t batch_size = slow5_batchsize;
    int32_t num_threads = slow5_threads;

    while(1){
        int flag_EOF = 0;
        db_t db = { 0 };
        db.mem_records = (char **) malloc(batch_size * sizeof *db.read_id);
        db.mem_bytes = (size_t *) malloc(batch_size * sizeof *db.read_id);

        int64_t record_count = 0;
        size_t bytes;
        char *mem;

        while (record_count < batch_size) {
            if (!(mem = (char *) slow5_get_next_mem(&bytes, sp))) {
                if (slow5_errno != SLOW5_ERR_EOF) {
                    throw std::runtime_error("Error in slow5_get_next_mem.");
                } else { //EOF file reached
                    flag_EOF = 1;
                    break;
                }
            } else {
                db.mem_records[record_count] = mem;
                db.mem_bytes[record_count] = bytes;
                record_count++;
            }
        }

        // Setup multithreading structures
        core_t core;
        core.num_thread = (num_threads > record_count) ? record_count : num_threads;
        if(record_count == 0){
            core.num_thread = 1;
        }
        core.fp = sp;
        core.m_device_ = m_device;

        db.n_batch = record_count;
        db.read_data_ptrs = std::vector<std::shared_ptr<Read>> (record_count);

        work_db(&core,&db,create_read_data);

        for (int64_t i = 0; i < record_count; i++) {
            m_read_sink.push_read(db.read_data_ptrs[i]);
            m_loaded_read_count++;
        }

        // Free everything
        free(db.mem_bytes);
        free(db.mem_records);

        if(flag_EOF == 1){
            break;
        }
    }
}


DataLoader::DataLoader(ReadSink& read_sink,
                       const std::string& device,
                       size_t num_worker_threads,
                       size_t max_reads,
                       int32_t slow5_threads,
                       int64_t slow5_batchsize)
        : m_read_sink(read_sink), m_device(device), m_num_worker_threads(num_worker_threads), slow5_threads(slow5_threads), slow5_batchsize(slow5_batchsize) {
    m_max_reads = max_reads == 0 ? std::numeric_limits<decltype(m_max_reads)>::max() : max_reads;
    assert(m_num_worker_threads > 0);
    static std::once_flag vbz_init_flag;
#ifdef USE_FAST5
    std::call_once(vbz_init_flag, vbz_register);
#endif
}

}  // namespace dorado
