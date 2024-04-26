#pragma once
#include <torch/torch.h>

namespace dorado::correction {

struct OverlapWindow {
    size_t overlap_idx = 0;
    int tstart = -1;
    int qstart = -1;
    int qend = -1;
    int cigar_start_idx = -1;
    int cigar_start_offset = -1;
    int cigar_end_idx = -1;
    int cigar_end_offset = -1;
    float accuracy = 0;
};

struct WindowFeatures {
    torch::Tensor bases;
    torch::Tensor quals;
    torch::Tensor indices;
    int length;
    std::vector<std::pair<int, int>> supported;
    std::vector<char> inferred_bases;
    int n_alns = 0;
    std::string read_name = "";
    int window_idx = -1;
};

}  // namespace dorado::correction
