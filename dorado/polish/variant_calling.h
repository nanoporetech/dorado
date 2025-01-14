#pragma once

#include "features/decoder_base.h"
#include "hts_io/FastxRandomReader.h"
#include "interval.h"
#include "sample.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

/**
 * \brief Type which holds the input data for variant calling. This includes _all_
 *          samples for the current batch of draft sequences and the inference results (logits) for those samples.
 *          The variant calling process will be responsible to group samples by draft sequence ID, etc.
 */
struct VariantCallingSample {
    Sample sample;
    at::Tensor logits;
};

struct Variant {
    int32_t seq_id = -1;
    int64_t pos = -1;
    std::string ref;
    std::string alt;
    std::string filter;
    std::unordered_map<std::string, std::string> info;
    std::string qual;
    std::unordered_map<std::string, std::string> genotype;
};

// Explicit full qualification of the Interval so it is not confused with the one from the IntervalTree library.
std::vector<Variant> call_variants(const dorado::polisher::Interval& region_batch,
                                   const std::vector<VariantCallingSample>& vc_input_data,
                                   const hts_io::FastxRandomReader& draft_reader,
                                   const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                                   const DecoderBase& decoder,
                                   const bool ambig_ref,
                                   const bool gvcf);

}  // namespace dorado::polisher
