#include "polish_impl.h"

#include "polish/interval.h"
#include "polish/polish_utils.h"
#include "polish/region.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/ssize.h"

#include <cxxpool.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <memory>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado::polisher {

void remove_deletions(ConsensusResult& cons) {
    if (std::size(cons.seq) != std::size(cons.quals)) {
        spdlog::error(
                "[remove_deletions] Sequence and quality string length mismatch! Not removing "
                "anything. seq.size = {}, quals.size = {}",
                std::size(cons.seq), std::size(cons.quals));
        return;
    }
    size_t n = 0;
    for (size_t j = 0; j < std::size(cons.seq); ++j) {
        if (cons.seq[j] == '*') {
            continue;
        }
        cons.seq[n] = cons.seq[j];
        cons.quals[n] = cons.quals[j];
        ++n;
    }
    cons.seq.resize(n);
    cons.quals.resize(n);
}

std::vector<ConsensusResult> stitch_sequence(
        const std::filesystem::path& in_draft_fn,
        const std::string& header,
        const std::vector<ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char) {
    const std::string draft = fetch_seq(in_draft_fn, header, 0, -1);
    const int64_t draft_len = dorado::ssize(draft);

    if (fill_gaps && std::empty(samples_for_seq)) {
        spdlog::debug(
                "Sequence '{}' of length {} has zero inferred samples. Copying contig verbatim "
                "from input.",
                header, std::size(draft));
        std::string dummy_quals(std::size(draft), '!');
        return {ConsensusResult{draft, std::move(dummy_quals)}};
    } else if (!fill_gaps && std::empty(samples_for_seq)) {
        spdlog::debug(
                "Sequence '{}' of length {} has zero inferred samples. NOT copying contig "
                "verbatim from input because fill_gaps == false.",
                header, std::size(draft));
        return {};
    }

    std::vector<ConsensusResult> ret;

    ConsensusResult result;
    result.draft_start = draft_len;
    result.draft_end = 0;

    // This is an inclusive coordinate.
    int64_t last_end = 0;
    for (size_t i = 0; i < std::size(samples_for_seq); ++i) {
        const int32_t sample_index = samples_for_seq[i].second;
        const ConsensusResult& sample_result = sample_results[sample_index];

        // Fill the gap with either the draft or a fill char.
        if (sample_result.draft_start > last_end) {
            if (fill_gaps) {
                const int64_t fill_len = sample_result.draft_start - last_end;
                result.seq += (fill_char) ? std::string(fill_len, *fill_char)
                                          : draft.substr(last_end, fill_len);
                result.quals += std::string(fill_len, '!');
                result.draft_start = std::min(result.draft_start, last_end);
                result.draft_end = std::max(result.draft_end, sample_result.draft_start);
            } else {
                if (!std::empty(result.seq)) {
                    ret.emplace_back(std::move(result));
                }
                result = {};
                result.draft_start = draft_len;
                result.draft_end = 0;
            }
        }

        // Splice a polished chunk.
        result.seq += sample_result.seq;
        result.quals += sample_result.quals;
        result.draft_start = std::min(result.draft_start, sample_result.draft_start);
        result.draft_end = std::max(result.draft_end, sample_result.draft_end);

        last_end = sample_result.draft_end;
    }

    // Add the back draft part (or fill char).
    if ((last_end < dorado::ssize(draft)) && fill_gaps) {
        const int64_t fill_len = draft_len - last_end;
        result.seq += (fill_char) ? std::string(fill_len, *fill_char) : draft.substr(last_end);
        result.quals += std::string(draft_len - last_end, '!');
        result.draft_start = std::min(result.draft_start, last_end);
        result.draft_end = std::max(result.draft_end, draft_len);
        if (!std::empty(result.seq)) {
            ret.emplace_back(std::move(result));
        }
    }

    spdlog::trace("[stitch_sequence] header = '{}', result.seq.size() = {}, final.", header,
                  std::size(result.seq));

    if (!std::empty(result.seq)) {
        ret.emplace_back(std::move(result));
    }

    return ret;
}

/**
 * \brief If the input sample coordinates (positions_major) have gaps,
 *          this function splits the sample on those gaps and produces
 *          one or more samples in the output.
 *          When possible, input data is moved to the output, and that is
 *          why the inpunt is not const.
 */
std::vector<Sample> split_sample_on_discontinuities(Sample& sample) {
    std::vector<Sample> results;

    const auto find_gaps = [](const std::vector<int64_t>& positions,
                              int64_t threshold) -> std::vector<int64_t> {
        std::vector<int64_t> ret;
        for (size_t i = 1; i < std::size(positions); ++i) {
            if ((positions[i] - positions[i - 1]) > threshold) {
                ret.emplace_back(i);
            }
        }
        return ret;
    };

    // Helper function to generate placeholder read IDs for read level models.
    const auto placeholder_read_ids = [](const int64_t n) {
        std::vector<std::string> placeholder_ids(n);
        for (int64_t i = 0; i < n; ++i) {
            placeholder_ids[i] = "__placeholder_" + std::to_string(i);
        }
        return placeholder_ids;
    };

    // Find gaps in data.
    const std::vector<int64_t> gaps = find_gaps(sample.positions_major, 1);

    // Reusable.
    const std::vector<std::string> placeholder_ids =
            placeholder_read_ids(dorado::ssize(sample.read_ids_left));

    if (std::empty(gaps)) {
        return {sample};

    } else {
        const int64_t num_positions = dorado::ssize(sample.positions_major);

        int64_t start = 0;
        for (int64_t n = 0; n < dorado::ssize(gaps); ++n) {
            const int64_t end = gaps[n];
            std::vector<int64_t> new_major_pos(std::begin(sample.positions_major) + start,
                                               std::begin(sample.positions_major) + end);
            std::vector<int64_t> new_minor_pos(std::begin(sample.positions_minor) + start,
                                               std::begin(sample.positions_minor) + end);

            std::vector<std::string> read_ids_left =
                    (n == 0) ? sample.read_ids_left : placeholder_ids;

            results.emplace_back(Sample{
                    sample.features.slice(0, start, end), std::move(new_major_pos),
                    std::move(new_minor_pos), sample.depth.slice(0, start, end), sample.seq_id,
                    sample.region_id, std::move(read_ids_left), placeholder_ids});
            start = end;
        }

        if (start < num_positions) {
            std::vector<int64_t> new_major_pos(std::begin(sample.positions_major) + start,
                                               std::end(sample.positions_major));
            std::vector<int64_t> new_minor_pos(std::begin(sample.positions_minor) + start,
                                               std::end(sample.positions_minor));
            results.emplace_back(Sample{sample.features.slice(0, start), std::move(new_major_pos),
                                        std::move(new_minor_pos), sample.depth.slice(0, start),
                                        sample.seq_id, sample.region_id, placeholder_ids,
                                        sample.read_ids_right});
        }
    }

    return results;
}

/**
 * \brief Takes an input sample and splits it bluntly into overlapping windows.
 *          Splitting is implemented to match Medaka, where a simple sliding window is used to create smaller samples.
 *          In case of a short trailing portion (shorter than chunk_len), a potentially large overlap is produced to
 *          cover this region instead of just outputing the small chunk.
 */
std::vector<Sample> split_samples(std::vector<Sample> samples,
                                  const int64_t chunk_len,
                                  const int64_t chunk_overlap) {
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    const auto create_chunk = [](const Sample& sample, const int64_t start, const int64_t end) {
        torch::Tensor new_features = sample.features.slice(0, start, end);
        std::vector<int64_t> new_major(std::begin(sample.positions_major) + start,
                                       std::begin(sample.positions_major) + end);
        std::vector<int64_t> new_minor(std::begin(sample.positions_minor) + start,
                                       std::begin(sample.positions_minor) + end);
        torch::Tensor new_depth = sample.depth.slice(0, start, end);
        return Sample{
                std::move(new_features),
                std::move(new_major),
                std::move(new_minor),
                std::move(new_depth),
                sample.seq_id,
                sample.region_id,
                {},
                {},
        };
    };

    std::vector<Sample> results;
    results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        if (sample_len <= chunk_len) {
            results.emplace_back(std::move(sample));
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        int64_t end = 0;
        for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
            end = start + chunk_len;
            results.emplace_back(create_chunk(sample, start, end));
        }

        // This will create a chunk with potentially large overlap.
        if (end < sample_len) {
            const int64_t start = sample_len - chunk_len;
            end = sample_len;
            results.emplace_back(create_chunk(sample, start, end));
        }
    }

    return results;
}

std::vector<Sample> encode_regions_in_parallel(
        std::vector<BamFile>& bam_handles,
        const EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const Window> windows,
        const int32_t num_threads) {
    // Worker function, each thread computes tensors for a set of windows assigned to it.
    const auto worker = [&](const int32_t thread_id, const int32_t start, const int32_t end,
                            std::vector<Sample>& results) {
        for (int32_t i = start; i < end; ++i) {
            const auto& window = windows[i];
            const std::string& name = draft_lens[window.seq_id].first;
            if (thread_id == 0) {
                spdlog::trace(
                        "[start = {}, end = {}] Encoding i = {}, region = "
                        "{}:{}-{} ({} %).",
                        start, end, i, name, window.start, window.end,
                        100.0 * static_cast<double>(i - start) / (end - start));
            }
            results[i] = encoder.encode_region(bam_handles[thread_id], name, window.start,
                                               window.end, window.seq_id);
        }
    };

    const std::vector<Interval> thread_chunks =
            compute_partitions(static_cast<int32_t>(std::size(windows)),
                               std::min(num_threads, static_cast<int32_t>(std::size(bam_handles))));

    spdlog::debug("Starting to encode regions for {} windows using {} threads.", std::size(windows),
                  std::size(thread_chunks));

    // Create the thread pool, futures and results.
    cxxpool::thread_pool pool{std::size(thread_chunks)};
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));

    std::vector<Sample> results(std::size(windows));

    // Add jobs to the pool.
    for (int32_t tid = 0; tid < dorado::ssize(thread_chunks); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, tid, chunk_start, chunk_end, std::ref(results)));
    }

    // Join and catch errors.
    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from encoding task: ") + e.what()};
    }

    return results;
}

std::pair<std::vector<Sample>, std::vector<TrimInfo>> merge_and_split_bam_regions_in_parallel(
        std::vector<Sample>& window_samples,
        const EncoderBase& encoder,
        const Span<const Window> bam_regions,
        const Span<const Interval> bam_region_intervals,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t window_interval_offset) {
    // Three tasks for this worker:
    //  1. Merge adjacent samples, which were split for efficiencly of computing the pileup.
    //  2. Check for discontinuities in any of the samples and split (gap in coverage).
    //  3. Split the merged samples into equally sized pieces which will be used for inference.

#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
    const auto debug_print_samples = [](std::ostream& os, const std::vector<Sample>& samples,
                                        int64_t start /* = 0*/, int64_t end /* = -1 */,
                                        int64_t debug_id /* = -1 */) {
        start = std::max<int64_t>(0, start);
        end = (end <= 0) ? static_cast<int64_t>(std::size(samples)) : end;

        for (int64_t i = start; i < end; ++i) {
            os << "[i = " << i << "] ";
            debug_print_sample(os, samples[i], 0, -1, i == debug_id);
            os << '\n';
        }
    };
#endif

    const auto worker = [&](const int32_t start, const int32_t end,
                            std::vector<std::vector<Sample>>& results_samples,
                            std::vector<std::vector<TrimInfo>>& results_trims) {
        for (int32_t bam_region_id = start; bam_region_id < end; ++bam_region_id) {
            Interval interval = bam_region_intervals[bam_region_id];
            interval.start -= window_interval_offset;
            interval.end -= window_interval_offset;

            spdlog::trace("- [bam_region_id = {}] (0) Before merging: interval = [{}, {}]",
                          bam_region_id, interval.start, interval.end);
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            std::cerr << "[merged_samples worker bam_region_id = " << bam_region_id
                      << "] Before merging. interval = [" << interval.start << ", " << interval.end
                      << ">:\n";
            std::cerr << "- [bam_region_id = " << bam_region_id << "] Input (window_samples):\n";
            debug_print_samples(std::cerr, window_samples, interval.start, interval.end, -1);
#endif

            std::vector<Sample> local_samples;

            // Split all samples on discontinuities.
            for (int32_t sample_id = interval.start; sample_id < interval.end; ++sample_id) {
                auto& sample = window_samples[sample_id];
                std::vector<Sample> disc_samples = split_sample_on_discontinuities(sample);
                local_samples.insert(std::end(local_samples),
                                     std::make_move_iterator(std::begin(disc_samples)),
                                     std::make_move_iterator(std::end(disc_samples)));
            }

            spdlog::trace(
                    "- [bam_region_id = {}] (1) After splitting on discontinuities: "
                    "local_samples = {}",
                    bam_region_id, std::size(local_samples));
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            std::cerr << "- [bam_region_id = " << bam_region_id
                      << "] After splitting on discontinuities (local_samples):\n";
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            local_samples = encoder.merge_adjacent_samples(local_samples);

            spdlog::trace("- [bam_region_id = {}] (2) After merging adjacent: local_samples = {}",
                          bam_region_id, std::size(local_samples));
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            std::cerr << "- [bam_region_id = " << bam_region_id
                      << "] After merging adjacent (local_samples):\n";
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            local_samples = split_samples(std::move(local_samples), window_len, window_overlap);

            spdlog::trace(
                    "- [bam_region_id = {}] (3) After splitting samples: local_samples = {}, "
                    "window_len = {}, window_overlap = {}",
                    bam_region_id, std::size(local_samples), window_len, window_overlap);
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            std::cerr << "- [bam_region_id = " << bam_region_id
                      << "] After splitting samples (local_samples):\n";
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            const Window& reg = bam_regions[bam_region_id];
            results_trims[bam_region_id] = trim_samples(
                    local_samples,
                    std::optional<Region>({reg.seq_id, reg.start_no_overlap, reg.end_no_overlap}));
            results_samples[bam_region_id] = std::move(local_samples);
        }
    };

    // InferenceInputs results;
    // results.samples.resize(std::size(bam_region_intervals));
    // results.trims.resize(std::size(bam_region_intervals));
    std::vector<std::vector<Sample>> merged_samples(std::size(bam_region_intervals));
    std::vector<std::vector<TrimInfo>> merged_trims(std::size(bam_region_intervals));

    // Process BAM windows in parallel.
    const std::vector<Interval> thread_chunks =
            compute_partitions(static_cast<int32_t>(std::size(bam_region_intervals)), num_threads);

    spdlog::trace("Starting to merge samples for {} BAM windows using {} threads.",
                  std::size(bam_region_intervals), std::size(thread_chunks));

    cxxpool::thread_pool pool{std::size(thread_chunks)};
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));

    for (int32_t tid = 0; tid < dorado::ssize(thread_chunks); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, chunk_start, chunk_end, std::ref(merged_samples),
                                       std::ref(merged_trims)));
    }

    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from merge-samples task: ") +
                                 e.what()};
    }

    // Flatten the samples.
    size_t num_samples = 0;
    for (const auto& vals : merged_samples) {
        num_samples += std::size(vals);
    }

    std::vector<Sample> results_samples;
    results_samples.reserve(num_samples);
    for (auto& vals : merged_samples) {
        results_samples.insert(std::end(results_samples), std::make_move_iterator(std::begin(vals)),
                               std::make_move_iterator(std::end(vals)));
    }

    std::vector<TrimInfo> results_trims;
    results_trims.reserve(num_samples);
    for (auto& vals : merged_trims) {
        results_trims.insert(std::end(results_trims), std::make_move_iterator(std::begin(vals)),
                             std::make_move_iterator(std::end(vals)));
    }

    return {results_samples, results_trims};
}

std::vector<Window> create_bam_regions(
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const int32_t bam_chunk_len,
        const int32_t window_overlap,
        const std::vector<std::string>& regions) {
    // Canonical case where each sequence is linearly split with an overlap.
    if (std::empty(regions)) {
        std::vector<Window> windows;
        for (int32_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
            const int64_t len = draft_lens[seq_id].second;
            const std::vector<Window> new_windows =
                    create_windows(seq_id, 0, len, len, bam_chunk_len, window_overlap);
            windows.reserve(std::size(windows) + std::size(new_windows));
            windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
        }
        return windows;
    } else {
        // Create windows for only this one region.
        std::unordered_map<std::string, int32_t> draft_ids;
        for (int32_t seq_id = 0; seq_id < dorado::ssize(draft_lens); ++seq_id) {
            draft_ids[draft_lens[seq_id].first] = seq_id;
        }

        std::vector<Window> windows;

        for (const auto& region_str : regions) {
            auto [region_name, region_start, region_end] = parse_region_string(region_str);

            spdlog::debug("Processing a custom region: '{}:{}-{}'.", region_name, region_start + 1,
                          region_end);

            const auto it = draft_ids.find(region_name);
            if (it == std::end(draft_ids)) {
                throw std::runtime_error(
                        "Sequence specified by custom region not found in input! Sequence name: " +
                        region_name);
            }
            const int32_t seq_id = it->second;
            const int64_t seq_length = draft_lens[seq_id].second;

            region_start = std::max<int64_t>(0, region_start);
            region_end = (region_end < 0) ? seq_length : std::min(seq_length, region_end);

            if (region_start >= region_end) {
                throw std::runtime_error{"Region coordinates not valid. Given: '" + region_str +
                                         "'. region_start = " + std::to_string(region_start) +
                                         ", region_end = " + std::to_string(region_end)};
            }

            // Split-up the custom region if it's too long.
            std::vector<Window> new_windows = create_windows(
                    seq_id, region_start, region_end, seq_length, bam_chunk_len, window_overlap);
            windows.reserve(std::size(windows) + std::size(new_windows));
            windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
        }

        return windows;
    }
}

}  // namespace dorado::polisher
