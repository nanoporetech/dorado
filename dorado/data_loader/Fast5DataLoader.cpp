#include "Fast5DataLoader.h"
#include "../read_pipeline/ReadPipeline.h"
#include <highfive/H5File.hpp>
#include <highfive/H5Easy.hpp>
#include <filesystem>

namespace {
void fixed_string_reader(HighFive::Attribute& attribute, std::string& target_str) {
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
}

void Fast5DataLoader::load_reads(const std::string& path) {
    if(!std::filesystem::exists(path)) {
        std::cerr << "Requested input path " << path << " does not exist!" << std::endl;
        m_read_sink.terminate();
        return;
    }
    if(!std::filesystem::is_directory(path)) {
        std::cerr << "Requested input path " << path << " is not a directory!" << std::endl;
        m_read_sink.terminate();
        return;
    }

    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        std::string ext = std::filesystem::path(entry).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
        if(ext == ".fast5") {
            load_reads_from_file(entry.path());
        }
    }
    m_read_sink.terminate();
    std::cerr << "> Loaded " << m_loaded_read_count << " reads" << std::endl;
}

void Fast5DataLoader::load_reads_from_file(const std::string& path) {

    // Read the file into a vector of torch tensors
    H5Easy::File file(path, H5Easy::File::ReadOnly);
    HighFive::Group reads = file.getGroup("/");
    int num_reads = reads.getNumberObjects();

    for (int i=0; i < num_reads; i++){
        auto read_id = reads.getObjectName(i);
        HighFive::Group read = reads.getGroup(read_id);

        // Fetch the digitisation parameters
        HighFive::Group channel_id_group = read.getGroup("channel_id");
        HighFive::Attribute digitisation_attr = channel_id_group.getAttribute("digitisation");
        HighFive::Attribute range_attr = channel_id_group.getAttribute("range");
        HighFive::Attribute offset_attr = channel_id_group.getAttribute("offset");
        // TODO: need to check type and if read as string then convert to int32_t
        // Reading as string currently not functioning
        // HighFive::Attribute channel_number_attr = channel_id_group.getAttribute("channel_number");
        // int32_t channel_number;
        float digitisation;
        digitisation_attr.read(digitisation);
        float range;
        range_attr.read(range);
        float offset;
        offset_attr.read(offset);

        HighFive::Group raw = read.getGroup("Raw");
        auto ds = raw.getDataSet("Signal");
        if (ds.getDataType().string() != "Integer16")
            throw std::runtime_error("Invalid FAST5 Signal data type of " + ds.getDataType().string());
        std::vector<int16_t> tmp;
        ds.read(tmp);
        std::vector<float> floatTmp(tmp.begin(), tmp.end());

        HighFive::Attribute mux_attr = raw.getAttribute("start_mux");
        HighFive::Attribute read_number_attr = raw.getAttribute("read_number");
        // TODO: Reading as string currently not functioning
        // HighFive::Attribute start_time_attr = raw.getAttribute("start_time");
        // std::string start_time;
        uint32_t mux;
        uint32_t read_number;
        mux_attr.read(mux);
        read_number_attr.read(read_number);
        std::string fast5_filename = std::filesystem::path(path).filename().string();

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto new_read = std::make_shared<Read>( 
            Read {
                .raw_data = torch::from_blob(floatTmp.data(), floatTmp.size(), options).clone().to(m_device),
                .digitisation = digitisation,
                .range = range,
                .offset = offset,
                .read_id = read_id,
                .attributes = Read::Attributes {
                    .mux = mux,
                    .read_number = read_number,
                    // .channel_number = channel_number,
                    // .start_time = start_time;
                    .fast5_filename = fast5_filename
                }
            }
        );

        m_read_sink.push_read(new_read);
        m_loaded_read_count++;
    }
}

Fast5DataLoader::Fast5DataLoader(ReadSink& read_sink, const std::string& device) :
    m_read_sink(read_sink),
    m_device(device){
}
