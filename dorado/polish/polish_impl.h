#pragma once

#include "polish/architectures/decoder_factory.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_factory.h"
#include "polish/consensus_result.h"
#include "polish/sample.h"
#include "polish/trim.h"
#include "utils/span.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::polisher {

struct Window {
    int32_t seq_id = -1;
    int64_t seq_length = 0;
    int64_t start = 0;
    int64_t end = 0;
    int32_t region_id = 0;
    int64_t start_no_overlap = 0;
    int64_t end_no_overlap = 0;
};

std::ostream& operator<<(std::ostream& os, const Window& w);

std::vector<Window> create_windows(const int32_t seq_id,
                                   const int64_t seq_start,
                                   const int64_t seq_end,
                                   const int64_t seq_len,
                                   const int32_t window_len,
                                   const int32_t window_overlap,
                                   const int32_t region_id);

ConsensusResult stitch_sequence(const std::filesystem::path& in_draft_fn,
                                const std::string& header,
                                const std::vector<ConsensusResult>& sample_results,
                                const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
                                [[maybe_unused]] const int32_t seq_id);

std::pair<std::vector<Sample>, std::vector<TrimInfo>> create_samples(
        std::vector<BamFile>& bam_handles,
        const EncoderBase& encoder,
        const std::vector<Window>& bam_regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t bam_subchunk_len);

std::vector<ConsensusResult> infer_samples_in_parallel(
        const std::vector<Sample>& in_samples,
        const std::vector<TrimInfo>& in_trims,
        const std::vector<std::shared_ptr<ModelTorchBase>>& models,
        const EncoderBase& encoder,
        const FeatureDecoder& decoder,
        const int32_t window_len,
        const int32_t batch_size);

std::vector<Window> create_bam_regions(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t bam_chunk_len,
        const int32_t window_overlap,
        const std::string& region_str);

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
        const int32_t window_overlap);

void remove_deletions(ConsensusResult& cons);

}  // namespace dorado::polisher
