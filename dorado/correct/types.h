#pragma once

#include <ATen/Tensor.h>

#include <filesystem>
#include <vector>

namespace dorado::correction {

struct OverlapWindow {
    int overlap_idx = -1;
    int tstart = -1;
    int qstart = -1;
    int qend = -1;
    int cigar_start_idx = -1;
    int cigar_start_offset = -1;
    int cigar_end_idx = -1;
    int cigar_end_offset = -1;
    float accuracy = 0.f;
};

struct WindowFeatures {
    at::Tensor bases;
    at::Tensor quals;
    at::Tensor indices;
    int length = 0;
    std::vector<std::pair<int, int>> supported;
    std::vector<char> inferred_bases;
    int n_alns = 0;
    std::string read_name = "";
    int window_idx = -1;

    size_t size() {
        size_t total = 0;
        total += bases.numel() * bases.element_size();
        total += quals.numel() * quals.element_size();
        total += indices.numel() * indices.element_size();
        total += supported.size() * sizeof(std::pair<int, int>);
        total += inferred_bases.size();
        total += sizeof(n_alns);
        total += read_name.length();
        total += sizeof(window_idx);
        return total;
    }
};

struct ModelConfig {
    int version;
    int window_size;
    std::string model_type;
    std::string weights_file;
    std::filesystem::path model_dir;
};

}  // namespace dorado::correction
