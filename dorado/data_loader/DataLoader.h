#pragma once
#include <string>

namespace dorado {

class ReadSink;

class DataLoader {
public:
    DataLoader(ReadSink& read_sink,
               const std::string& device,
               size_t num_worker_threads,
               size_t max_reads = 0);
    void load_reads(const std::string& path);

private:
    void load_fast5_reads_from_file(const std::string& path);
#ifdef USE_POD5
    void load_pod5_reads_from_file(const std::string& path);
#endif
    void load_slow5_reads_from_file(const std::string& path);
    ReadSink& m_read_sink;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
    int64_t slow5_batch = 4000;
    int32_t slow5_threads = 8;
    size_t m_num_worker_threads{1};
    size_t m_max_reads{0};
};

}  // namespace dorado
