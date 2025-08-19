
#include "secondary/consensus/sample_collate_utils.h"

#include "utils/ssize.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace dorado::secondary {

std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples) {
    if (std::empty(buffered_samples)) {
        return {};
    }

    // The +1 is for the batch size dimension.
    std::vector<int64_t> ret(1 + std::size(buffered_samples.front().features.sizes()), 0);

    // For each features tensor, find the maximum sizes for all dimensions.
    for (int64_t i = 0; i < dorado::ssize(buffered_samples); ++i) {
        const secondary::Sample& sample = buffered_samples[i];

        if (!sample.features.defined()) {
            throw std::runtime_error{
                    "Buffered sample.features tensor not defined in compute_collated_padded_shape! "
                    "Buffered sample ID: " +
                    std::to_string(i)};
        }

        // Sanity check, though this should never happen.
        if ((std::size(sample.features.sizes()) + 1) != std::size(ret)) {
            throw std::runtime_error{"Input tensors are not all of the same dimensionality!"};
        }

        // Increase the batch size.
        ++ret[0];

        // Find max on every dimension (mock padding).
        for (int64_t j = 0; j < dorado::ssize(sample.features.sizes()); ++j) {
            ret[j + 1] = std::max(ret[j + 1], sample.features.size(j));
        }
    }

    return ret;
}

std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples,
                                                   const Sample& new_sample) {
    if (!new_sample.features.defined()) {
        throw std::runtime_error{
                "Tensor new_sample.features not defined in compute_collated_padded_shape!"};
    }

    // This has 1 extra dimension compared to all samples, for the batch size.
    std::vector<int64_t> ret = compute_collated_padded_shape(buffered_samples);

    if (std::empty(ret)) {
        ret = std::vector<int64_t>(1 + std::size(new_sample.features.sizes()), 0);
    }

    if (std::size(ret) != (std::size(new_sample.features.sizes()) + 1)) {
        throw std::runtime_error{
                "The new_sample input tensor shape does not match the tensors in "
                "buffered_samples!"};
    }

    // Expand the batch.
    ++ret[0];

    // Find max on every dimension (mock padding).
    for (int64_t j = 0; j < dorado::ssize(new_sample.features.sizes()); ++j) {
        ret[j + 1] = std::max(ret[j + 1], new_sample.features.size(j));
    }

    return ret;
}

}  // namespace dorado::secondary
