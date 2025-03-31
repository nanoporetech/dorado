#pragma once

#include "features/decoder_base.h"
#include "hts_io/FastxRandomReader.h"
#include "polish_stats.h"
#include "sample.h"
#include "secondary/interval.h"
#include "secondary/variant.h"
#include "variant_calling_sample.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace dorado::polisher {

// Explicit full qualification of the Interval so it is not confused with the one from the IntervalTree library.
std::vector<secondary::Variant> call_variants(
        const secondary::Interval& region_batch,
        const std::vector<VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const DecoderBase& decoder,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        PolishStats& polish_stats);

secondary::Variant normalize_variant(const std::string_view ref_with_gaps,
                                     const std::vector<std::string_view>& cons_seqs_with_gaps,
                                     const std::vector<int64_t>& positions_major,
                                     const std::vector<int64_t>& positions_minor,
                                     const secondary::Variant& variant);

std::vector<VariantCallingSample> merge_vc_samples(
        const std::vector<VariantCallingSample>& vc_samples);

std::vector<secondary::Variant> decode_variants(const DecoderBase& decoder,
                                                const VariantCallingSample& vc_sample,
                                                const std::string& draft,
                                                const bool ambig_ref,
                                                const bool gvcf);

}  // namespace dorado::polisher
