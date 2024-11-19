#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct Sample {
    torch::Tensor features;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    torch::Tensor depth;
    int32_t seq_id = -1;
    int32_t region_id = -1;

    int64_t start() const { return (std::empty(positions_major) ? -1 : (positions_major.front())); }

    int64_t end() const {
        return (std::empty(positions_major) ? -1 : (positions_major.back() + 1));
    }

    std::pair<int64_t, int64_t> get_position(const int64_t idx) const {
        if ((idx < 0) || (idx >= static_cast<int64_t>(std::size(positions_major)))) {
            return {-1, -1};
        }
        return {positions_major[idx], positions_minor[idx]};
    }

    std::pair<int64_t, int64_t> get_last_position() const {
        return get_position(static_cast<int64_t>(std::size(positions_major)) - 1);
    }
};

inline void debug_print_sample(std::ostream& os,
                               const polisher::Sample& sample,
                               int64_t start /*= 0*/,
                               int64_t end /*= -1 */,
                               bool debug /*= false */) {
    const int64_t len = static_cast<int64_t>(std::size(sample.positions_major));
    start = std::max<int64_t>(0, start);
    end = (end <= 0) ? len : end;

    os << "sample.positions = " << sample.start() << " - " << sample.end()
       << " , dist = " << (sample.end() - sample.start()) << " , tensor = [";
    os.flush();
    for (int64_t k = start; k < std::min<int64_t>(start + 3, len); ++k) {
        os << "(" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ") ";
        os.flush();
    }
    os << " ...";
    os.flush();
    for (int64_t k = std::max<int64_t>(0, end - 3); k < end; ++k) {
        os << " (" << sample.positions_major[k] << ", " << sample.positions_minor[k] << ")";
        os.flush();
    }
    os << "] , size = " << std::size(sample.positions_major);
    os.flush();

    if (debug) {
        const auto depth = sample.depth.slice(/*dim=*/0, /*start=*/0);
        for (int64_t k = 0; k < len; ++k) {
            os << "[k = " << k << "] pos = (" << sample.positions_major[k] << ", "
               << sample.positions_minor[k] << "), depth = " << depth[k].item<float>() << "\n";
            os.flush();
        }
    }
}

inline std::ostream& operator<<(std::ostream& os, const Sample& sample) {
    debug_print_sample(os, sample, 0, -1, false);
    return os;
}

}  // namespace dorado::polisher
