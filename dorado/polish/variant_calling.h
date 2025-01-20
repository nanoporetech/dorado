#pragma once

#include "features/decoder_base.h"
#include "hts_io/FastxRandomReader.h"
#include "interval.h"
#include "polish_stats.h"
#include "sample.h"
#include "variant.h"
#include "variant_calling_sample.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

// Explicit full qualification of the Interval so it is not confused with the one from the IntervalTree library.
std::vector<Variant> call_variants(
        const dorado::polisher::Interval& region_batch,
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const DecoderBase& decoder,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        PolishStats& polish_stats);

}  // namespace dorado::polisher
