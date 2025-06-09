#pragma once

#include <ATen/Tensor.h>

#include <filesystem>
#include <iosfwd>
#include <vector>

namespace dorado::correction {

// clang-format off
struct OverlapWindow {
    // CorrectionAlignments overlap vector index
    int overlap_idx = -1;
    int win_tstart = -1;            // Window target start position.
    int win_tend = -1;              // Window target end position.
    int tstart = -1;                // Absolute alignment target start position for this window.
    int tend = -1;                  // Absolute alignment target end position for this window.
    int qstart = -1;                // Absolute alignment query start position for this window. (Does not include the absolute qstart from the alignment due to legacy reasons).
    int qend = -1;                  // Absolute alignment query end position for this window. (Does not include the absolute qstart from the alignment due to legacy reasons).
    int cigar_start_idx = -1;       // CIGAR start operation for the alignment within this window.
    int cigar_start_offset = -1;    // Offset within the CIGAR start operation where the alignment for this window begins.
    int cigar_end_idx = -1;         // CIGAR end operation for the alignment within this window. Can be inclusive or exclusive, depending on cigar_end_offset.
    int cigar_end_offset = -1;      // Offset within the CIGAR end operation where the alignment for this window ends. Non-inclusive.
    float accuracy = 0.f;           // Alignment accuracy within this window.
    int columns = 0;                // Number of pileup columns that the alignment in this window requires (M+EQ+X+I+D).
};
// clang-format on

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

std::ostream& operator<<(std::ostream& os, const OverlapWindow& ovl);

std::ostream& operator<<(std::ostream& os, const WindowFeatures& wf);

bool operator==(const OverlapWindow& lhs, const OverlapWindow& rhs);

}  // namespace dorado::correction