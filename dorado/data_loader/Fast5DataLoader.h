#pragma once
#include <string>
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include "../read_pipeline/ReadPipeline.h"


class Fast5DataLoader {
public:
    Fast5DataLoader(ReadSink& read_sink, const std::string& device);
    void load_reads(const std::string& path);

private:
    void load_reads_from_file(const std::string& path);
    ReadSink& m_read_sink; // Where should the loaded reads go?
    size_t m_loaded_read_count{0};
    std::string m_device;
};

