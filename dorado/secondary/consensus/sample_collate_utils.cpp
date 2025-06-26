
#include "secondary/consensus/sample_collate_utils.h"

#include "utils/ssize.h"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace dorado::secondary {

/**
 * \brief Given an existing vector of samples (a batch) and a new sample which is not yet added to the batch, this function
 *          computes the shape of the would-be batch tensor.
 *          The shape assumes padding in all dimensions.
 *          This is needed to estimate the memory consumption for a batch.
 */
std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples,
                                                   const Sample& new_sample) {
    // The +1 is for the batch size dimension.
    std::vector<int64_t> ret(1 + std::size(new_sample.features.sizes()), 0);

    if (!new_sample.features.defined()) {
        throw std::runtime_error{
                "Tensor new_sample.features not defined in compute_collated_padded_shape!"};
    }

    // For each features tensor, find the maximum sizes for all dimensions.
    for (int64_t i = 0; i < (dorado::ssize(buffered_samples) + 1); ++i) {
        const secondary::Sample& sample =
                (i < dorado::ssize(buffered_samples)) ? buffered_samples[i] : new_sample;

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

}  // namespace dorado::secondary
