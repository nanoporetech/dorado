#include "trim.h"

#include <algorithm>

namespace dorado::utils {
int trim(torch::Tensor signal,
         float threshold,
         int max_samples,
         int window_size,
         int min_elements) {
    int min_trim = 10;
    bool seen_peak = false;
    int num_samples = std::min(max_samples, static_cast<int>(signal.size(0)) - min_trim);
    int num_windows = num_samples / window_size;

    for (int pos = 0; pos < num_windows; pos++) {
        int start = pos * window_size + min_trim;
        int end = start + window_size;

        auto window = signal.index({torch::indexing::Slice(start, end)});
        auto elements = window > threshold;

        if ((elements.sum().item<int>() > min_elements) || seen_peak) {
            seen_peak = true;
            if (window[-1].item<float>() > threshold) {
                continue;
            }
            if (end >= num_samples) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}

}  // namespace dorado::utils
