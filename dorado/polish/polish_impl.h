#pragma once

#include "polish/architectures/decoder_factory.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_factory.h"
#include "polish/consensus_result.h"
#include "polish/sample.h"
#include "polish/trim.h"

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

struct Interval {
    int32_t start = 0;
    int32_t end = 0;
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
        const polisher::EncoderBase& encoder,
        const std::vector<Window>& bam_regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t bam_subchunk_len);

std::vector<ConsensusResult> process_samples_in_parallel(
        const std::vector<Sample>& in_samples,
        const std::vector<polisher::TrimInfo>& in_trims,
        const std::vector<std::shared_ptr<TorchModel>>& models,
        const polisher::EncoderBase& encoder,
        const polisher::FeatureDecoder& decoder,
        const int32_t window_len,
        const int32_t batch_size);

std::vector<Window> create_bam_regions(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t bam_chunk_len,
        const int32_t window_overlap,
        const std::string& region_str);

void remove_deletions(polisher::ConsensusResult& cons);

}  // namespace dorado::polisher
