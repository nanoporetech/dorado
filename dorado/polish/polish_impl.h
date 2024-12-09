#pragma once

#include "polish/architectures/decoder_factory.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_factory.h"
#include "polish/consensus_result.h"
#include "polish/sample.h"
#include "polish/trim.h"
#include "polish/window.h"
#include "utils/span.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct InferenceData {
    std::vector<Sample> samples;
    std::vector<TrimInfo> trims;
};

struct DecodeData {
    std::vector<Sample> samples;
    torch::Tensor logits;
    std::vector<TrimInfo> trims;
};

std::vector<ConsensusResult> stitch_sequence(
        const std::filesystem::path& in_draft_fn,
        const std::string& header,
        const std::vector<ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char,
        [[maybe_unused]] const int32_t seq_id);

std::vector<Window> create_bam_regions(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t bam_chunk_len,
        const int32_t window_overlap,
        const std::vector<std::string>& regions);

std::vector<Sample> encode_regions_in_parallel(
        std::vector<BamFile>& bam_handles,
        const EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const Window> windows,
        const int32_t num_threads);

std::pair<std::vector<Sample>, std::vector<TrimInfo>> merge_and_split_bam_regions_in_parallel(
        std::vector<Sample>& window_samples,
        const EncoderBase& encoder,
        const Span<const Window> bam_regions,
        const Span<const Interval> bam_region_intervals,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t window_interval_offset);

void remove_deletions(ConsensusResult& cons);

}  // namespace dorado::polisher
