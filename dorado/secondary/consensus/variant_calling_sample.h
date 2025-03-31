#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

namespace dorado::secondary {

class DecoderBase;

/**
 * \brief Type which holds the input data for variant calling. This includes _all_
 *          samples for the current batch of draft sequences and the inference results (logits) for those samples.
 *          The variant calling process will be responsible to group samples by draft sequence ID, etc.
 */
struct VariantCallingSample {
    int32_t seq_id = -1;
    std::vector<int64_t> positions_major;
    std::vector<int64_t> positions_minor;
    at::Tensor logits;

    void validate() const;
    int64_t start() const;
    int64_t end() const;
};

std::ostream& operator<<(std::ostream& os, const VariantCallingSample& vc_sample);

bool operator==(const VariantCallingSample& lhs, const VariantCallingSample& rhs);

VariantCallingSample slice_vc_sample(const VariantCallingSample& vc_sample,
                                     int64_t idx_start,
                                     int64_t idx_end);

std::vector<VariantCallingSample> merge_vc_samples(
        const std::vector<VariantCallingSample>& vc_samples);

std::vector<VariantCallingSample> join_samples(const std::vector<VariantCallingSample>& vc_samples,
                                               const std::string& draft,
                                               const DecoderBase& decoder);

std::vector<VariantCallingSample> trim_vc_samples(
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::pair<int64_t, int32_t>>& group);

}  // namespace dorado::secondary
