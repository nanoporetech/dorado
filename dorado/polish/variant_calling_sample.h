#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace dorado::polisher {

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

}  // namespace dorado::polisher
