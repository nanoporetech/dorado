#pragma once

#include "sample.h"

#include <cstdint>
#include <vector>

namespace dorado::secondary {

/**
 * \brief Given an existing vector of samples (a batch), this function
 *          computes the shape of the would-be batch tensor if samples were collated.
 *          The shape assumes padding in all dimensions.
 *          This is needed to estimate the memory consumption for a batch.
 */
std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples);

/**
 * \brief Given an existing vector of samples (a batch) and a new sample which is not yet added to the batch, this function
 *          computes the shape of the would-be batch tensor.
 *          The shape assumes padding in all dimensions.
 *          This is needed to estimate the memory consumption for a batch.
 */
std::vector<int64_t> compute_collated_padded_shape(const std::vector<Sample>& buffered_samples,
                                                   const Sample& new_sample);

}  // namespace dorado::secondary
