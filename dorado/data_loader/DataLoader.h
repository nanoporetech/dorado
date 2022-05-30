#pragma once
#include <string>

class ReadSink;

class DataLoader {
public:
    DataLoader(ReadSink& read_sink, const std::string& device);
    void load_reads(const std::string& path);

private:
    void load_fast5_reads_from_file(const std::string& path);
    void load_pod5_reads_from_file(const std::string& path);
    ReadSink& m_read_sink;  // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
};
