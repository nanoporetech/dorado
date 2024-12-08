#include "polish_impl.h"

#include "polish/interval.h"
#include "polish/polish_utils.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/ssize.h"
#include "utils/timer_high_res.h"

#include <cxxpool.h>
#include <htslib/faidx.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado::polisher {

std::ostream& operator<<(std::ostream& os, const Window& w) {
    os << "seq_id = " << w.seq_id << ", start = " << w.start << ", end = " << w.end
       << ", seq_length = " << w.seq_length
       << ", region_id = " << w.region_id;  // << ", window_id = " << w.window_id;
    return os;
}

/**
 * \brief Linearly splits sequence lengths into windows. It also returns the backward mapping of which
 *          windows correspond to which sequences, needed for stitching.
 */
std::vector<Window> create_windows(const int32_t seq_id,
                                   const int64_t seq_start,
                                   const int64_t seq_end,
                                   const int64_t seq_len,
                                   const int32_t window_len,
                                   const int32_t window_overlap,
                                   const int32_t region_id) {
    if (window_overlap >= window_len) {
        spdlog::error(
                "The window overlap cannot be larger than the window size! window_len = {}, "
                "window_overlap = {}\n",
                window_len, window_overlap);
        return {};
    }

    std::vector<Window> ret;
    const int64_t length = seq_end - seq_start;

    const int32_t num_windows =
            static_cast<int32_t>(std::ceil(static_cast<double>(length) / window_len));

    ret.reserve(num_windows);

    int32_t win_id = 0;
    for (int64_t start = seq_start; start < seq_end;
         start += (window_len - window_overlap), ++win_id) {
        const int64_t end = std::min(seq_end, start + window_len);
        const int64_t start_no_overlap =
                (start == seq_start) ? start : std::min<int64_t>(start + window_overlap, seq_end);

        ret.emplace_back(Window{
                seq_id,
                seq_len,
                start,
                end,
                region_id,
                start_no_overlap,
                end,
        });

        if (end == seq_end) {
            break;
        }
    }

    return ret;
}

std::string fetch_seq(const std::filesystem::path& index_fn,
                      const std::string& seq_name,
                      int32_t start,
                      int32_t end) {
    faidx_t* fai = fai_load(index_fn.string().c_str());
    if (!fai) {
        spdlog::error("Failed to load index for file: '{}'.", index_fn.string());
        return {};
    }

    const int32_t seq_len = faidx_seq_len(fai, seq_name.c_str());

    start = std::max(start, 0);
    end = (end < 0) ? seq_len : std::min(end, seq_len);

    int32_t temp_seq_len = 0;
    char* seq = faidx_fetch_seq(fai, seq_name.c_str(), start, end - 1, &temp_seq_len);

    if (end <= start) {
        spdlog::error(
                "Cannot load sequence because end <= start! seq_name = {}, start = {}, end = {}.",
                seq_name, start, end);
        return {};
    }

    if (temp_seq_len != (end - start)) {
        spdlog::error(
                "Loaded sequence length does not match the specified interval! seq_name = {}, "
                "start = {}, end = {}, loaded len = {}.",
                seq_name, start, end, temp_seq_len);
        return {};
    }

    std::string ret;
    if (seq) {
        ret = std::string(seq, temp_seq_len);
        free(seq);
    }

    fai_destroy(fai);

    return ret;
}

#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
void debug_print_samples(std::ostream& os,
                         const std::vector<polisher::Sample>& samples,
                         int64_t start /* = 0*/,
                         int64_t end /* = -1 */,
                         int64_t debug_id /* = -1 */) {
    start = std::max<int64_t>(0, start);
    end = (end <= 0) ? static_cast<int64_t>(std::size(samples)) : end;

    for (int64_t i = start; i < end; ++i) {
        os << "[i = " << i << "] ";
        polisher::debug_print_sample(os, samples[i], 0, -1, i == debug_id);
        os << '\n';
    }
}
#endif

void remove_deletions(polisher::ConsensusResult& cons) {
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

std::vector<polisher::ConsensusResult> stitch_sequence(
        const std::filesystem::path& in_draft_fn,
        const std::string& header,
        const std::vector<polisher::ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char,
        [[maybe_unused]] const int32_t seq_id) {
    const std::string draft = fetch_seq(in_draft_fn, header, 0, -1);
    const int64_t draft_len = dorado::ssize(draft);

    if (std::empty(samples_for_seq) && fill_gaps) {
        if (fill_gaps) {
            spdlog::debug(
                    "Sequence '{}' of length {} has zero inferred samples. Copying contig verbatim "
                    "from input.",
                    header, std::size(draft));
            std::string dummy_quals(std::size(draft), '!');
            return {polisher::ConsensusResult{draft, std::move(dummy_quals)}};
        } else {
            spdlog::debug(
                    "Sequence '{}' of length {} has zero inferred samples. NOT copying contig "
                    "verbatim "
                    "from input because fill_gaps == false.",
                    header, std::size(draft));
            return {};
        }
    }

    std::vector<polisher::ConsensusResult> ret;
    polisher::ConsensusResult result;
    result.draft_start = draft_len;
    result.draft_end = 0;

#ifdef DEBUG_POLISH_DUMP_SEQ_PIECES
    std::ofstream ofs("debug.seq_id_" + std::to_string(seq_id) + ".fasta");
#endif

    // This is an inclusive coordinate. If it was 0, then adding front draft chunk would miss 1 base.
    int64_t last_end = 0;
    for (size_t i = 0; i < std::size(samples_for_seq); ++i) {
        const int32_t sample_index = samples_for_seq[i].second;
        const polisher::ConsensusResult& sample_result = sample_results[sample_index];

        if (sample_result.draft_start > last_end) {
            if (fill_gaps) {
                spdlog::trace(
                        "[stitch_sequence] Gap between polished chunks. header = '{}', "
                        "result.seq.size() = {}, fill_char = {}, draft region: "
                        "[{}, {}], using fill_char: {}",
                        header, std::size(result.seq),
                        ((fill_char) ? std::string(1, *fill_char) : "nullopt"), last_end,
                        sample_result.draft_start, fill_char ? "true" : "false");

                const int64_t fill_len = sample_result.draft_start - last_end;
                result.seq += (fill_char) ? std::string(fill_len, *fill_char)
                                          : draft.substr(last_end, fill_len);
                result.quals += std::string(fill_len, '!');
                result.draft_start = std::min(result.draft_start, last_end);
                result.draft_end = std::max(result.draft_end, sample_result.draft_start);
            } else {
                spdlog::trace(
                        "[stitch_sequence] Gap between polished chunks. header = '{}', "
                        "result.seq.size() = {}, draft region: "
                        "[{}, {}]. Dumping the current polished subchunk.",
                        header, std::size(result.seq), last_end, sample_result.draft_start);

                if (!std::empty(result.seq)) {
                    ret.emplace_back(std::move(result));
                }
                result = {};
                result.draft_start = draft_len;
                result.draft_end = 0;
            }
        }

        spdlog::trace(
                "[stitch_sequence] header = '{}', result.seq.size() = {}, adding consensus chunk "
                "sample_result.seq.size() = {}",
                header, std::size(result.seq), std::size(sample_result.seq));

        result.seq += sample_result.seq;
        result.quals += sample_result.quals;
        result.draft_start = std::min(result.draft_start, sample_result.draft_start);
        result.draft_end = std::max(result.draft_end, sample_result.draft_end);

        last_end = sample_result.draft_end;
    }

    // Add the back draft part.
    if (last_end < dorado::ssize(draft)) {
        if (fill_gaps) {
            spdlog::trace(
                    "[stitch_sequence] Trailing gap. header = '{}', result.seq.size() = {}, "
                    "fill_char = {}, draft region: "
                    "[{}, {}], using fill_char: {}",
                    header, std::size(result.seq),
                    ((fill_char) ? std::string(1, *fill_char) : "nullopt"), last_end, draft_len,
                    fill_char ? "true" : "false");

            const int64_t fill_len = draft_len - last_end;
            result.seq += (fill_char) ? std::string(fill_len, *fill_char) : draft.substr(last_end);
            result.quals += std::string(draft_len - last_end, '!');
            result.draft_start = std::min(result.draft_start, last_end);
            result.draft_end = std::max(result.draft_end, draft_len);
            if (!std::empty(result.seq)) {
                ret.emplace_back(std::move(result));
            }
        } else {
            spdlog::trace(
                    "[stitch_sequence] Trailing gap. header = '{}', result.seq.size() = {}, draft "
                    "region: "
                    "[{}, {}]. Dumping the current polished subchunk.",
                    header, std::size(result.seq), last_end, draft_len);
        }
    }

    if (!std::empty(result.seq)) {
        ret.emplace_back(std::move(result));
    }

    spdlog::trace("[stitch_sequence] header = '{}', result.seq.size() = {}, final.", header,
                  std::size(result.seq));

    return ret;
}

std::vector<polisher::Sample> split_sample_on_discontinuities(polisher::Sample& sample) {
    std::vector<polisher::Sample> results;

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

    // Helper function to generate placeholder read IDs
    const auto placeholder_read_ids = [](const int64_t n) {
        std::vector<std::string> placeholder_ids(n);
        for (int64_t i = 0; i < n; ++i) {
            placeholder_ids[i] = "__placeholder_" + std::to_string(i);
        }
        return placeholder_ids;
    };

    // for (auto& data : pileups) {
    const std::vector<int64_t> gaps = find_gaps(sample.positions_major, 1);

    const std::vector<std::string> placeholder_ids =
            placeholder_read_ids(dorado::ssize(sample.read_ids_left));

    if (std::empty(gaps)) {
        return {sample};

    } else {
        const int64_t num_positions = dorado::ssize(sample.positions_major);

        int64_t start = 0;
        for (int64_t n = 0; n < dorado::ssize(gaps); ++n) {
            const int64_t i = gaps[n];
            std::vector<int64_t> new_major_pos(sample.positions_major.begin() + start,
                                               sample.positions_major.begin() + i);
            std::vector<int64_t> new_minor_pos(sample.positions_minor.begin() + start,
                                               sample.positions_minor.begin() + i);

            std::vector<std::string> read_ids_left =
                    (n == 0) ? sample.read_ids_left : placeholder_ids;

            results.emplace_back(polisher::Sample{
                    sample.features.slice(0, start, i), std::move(new_major_pos),
                    std::move(new_minor_pos), sample.depth.slice(0, start, i), sample.seq_id,
                    sample.region_id, std::move(read_ids_left), placeholder_ids});
            start = i;
        }

        if (start < num_positions) {
            std::vector<int64_t> new_major_pos(sample.positions_major.begin() + start,
                                               sample.positions_major.end());
            std::vector<int64_t> new_minor_pos(sample.positions_minor.begin() + start,
                                               sample.positions_minor.end());
            results.emplace_back(polisher::Sample{
                    sample.features.slice(0, start), std::move(new_major_pos),
                    std::move(new_minor_pos), sample.depth.slice(0, start), sample.seq_id,
                    sample.region_id, placeholder_ids, sample.read_ids_right});
        }
    }

    return results;
}

/**
 * \brief Takes an input sample and splits it bluntly if it has too many positions. This can happen when
 *          there are many long insertions in an input window, and can easily cause out-of-memory issues on the GPU
 *          if the sample is not split.
 *          Splitting is implemented to match Medaka, where a simple sliding window is used to create smaller samples.
 *          In case of a smalle trailing portion (smaller than chunk_len), a potentially large overlap is produced to
 *          cover this region instead of just outputing the small chunk.
 */
std::vector<polisher::Sample> split_samples(std::vector<polisher::Sample> samples,
                                            const int64_t chunk_len,
                                            const int64_t chunk_overlap) {
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    const auto create_chunk = [](const polisher::Sample& sample, const int64_t start,
                                 const int64_t end) {
        torch::Tensor new_features = sample.features.slice(0, start, end);
        std::vector<int64_t> new_major(std::begin(sample.positions_major) + start,
                                       std::begin(sample.positions_major) + end);
        std::vector<int64_t> new_minor(std::begin(sample.positions_minor) + start,
                                       std::begin(sample.positions_minor) + end);
        torch::Tensor new_depth = sample.depth.slice(0, start, end);
        return polisher::Sample{
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

    std::vector<polisher::Sample> results;
    results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        if (sample_len <= chunk_len) {
            results.emplace_back(std::move(sample));
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        spdlog::trace("[split_samples] sample_len = {}, features.shape = {}, step = {}", sample_len,
                      tensor_shape_as_string(sample.features), step);

        int64_t end = 0;
        for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
            end = start + chunk_len;
            spdlog::trace("[split_samples]     - creating chunk: start = {}, end = {}", start, end);
            results.emplace_back(create_chunk(sample, start, end));
        }

        // This will create a chunk with potentially large overlap.
        if (end < sample_len) {
            const int64_t start = sample_len - chunk_len;
            end = sample_len;
            spdlog::trace("[split_samples]     - creating end chunk: start = {}, end = {}", start,
                          end);
            results.emplace_back(create_chunk(sample, start, end));
        }
    }

    spdlog::trace("[split_samples]     - done, results.size() = {}", std::size(results));

    return results;
}

std::tuple<std::string, int64_t, int64_t> parse_region_string(const std::string& region) {
    const size_t colon_pos = region.find(':');
    if (colon_pos == std::string::npos) {
        return {region, -1, -1};
    }

    std::string name = region.substr(0, colon_pos);

    if ((colon_pos + 1) == std::size(region)) {
        return {std::move(name), -1, -1};
    }

    size_t dash_pos = region.find('-', colon_pos + 1);
    dash_pos = (dash_pos == std::string::npos) ? std::size(region) : dash_pos;
    const int64_t start =
            ((dash_pos - colon_pos - 1) == 0)
                    ? -1
                    : std::stoll(region.substr(colon_pos + 1, dash_pos - colon_pos - 1)) - 1;
    const int64_t end =
            ((dash_pos + 1) < std::size(region)) ? std::stoll(region.substr(dash_pos + 1)) : -1;

    return {std::move(name), start, end};
}

std::vector<Sample> encode_regions_in_parallel(
        std::vector<BamFile>& bam_handles,
        const polisher::EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const Window> windows,
        const int32_t num_threads) {
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
            compute_chunks(static_cast<int32_t>(std::size(windows)),
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

    const auto worker = [&](const int32_t start, const int32_t end,
                            std::vector<std::vector<polisher::Sample>>& results_samples,
                            std::vector<std::vector<polisher::TrimInfo>>& results_trims) {
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

            std::vector<polisher::Sample> local_samples;

            // Split all samples on discontinuities.
            for (int32_t sample_id = interval.start; sample_id < interval.end; ++sample_id) {
                auto& sample = window_samples[sample_id];
                std::vector<polisher::Sample> disc_samples =
                        split_sample_on_discontinuities(sample);
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
            results_trims[bam_region_id] = polisher::trim_samples(
                    local_samples, {reg.seq_id, reg.start_no_overlap, reg.end_no_overlap});
            results_samples[bam_region_id] = std::move(local_samples);
        }
    };

    // InferenceInputs results;
    // results.samples.resize(std::size(bam_region_intervals));
    // results.trims.resize(std::size(bam_region_intervals));
    std::vector<std::vector<polisher::Sample>> merged_samples(std::size(bam_region_intervals));
    std::vector<std::vector<polisher::TrimInfo>> merged_trims(std::size(bam_region_intervals));

    // Process BAM windows in parallel.
    const std::vector<Interval> thread_chunks =
            compute_chunks(static_cast<int32_t>(std::size(bam_region_intervals)), num_threads);

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

    std::vector<polisher::Sample> results_samples;
    results_samples.reserve(num_samples);
    for (auto& vals : merged_samples) {
        results_samples.insert(std::end(results_samples), std::make_move_iterator(std::begin(vals)),
                               std::make_move_iterator(std::end(vals)));
    }

    std::vector<polisher::TrimInfo> results_trims;
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
                    create_windows(seq_id, 0, len, len, bam_chunk_len, window_overlap, -1);
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
                        "Sequence provided by custom region not found in input! Sequence name: " +
                        region_name);
            }
            const int32_t seq_id = it->second;
            const int64_t seq_length = draft_lens[seq_id].second;

            region_start = std::max<int64_t>(0, region_start);
            region_end = std::min(seq_length, region_end);

            // Split-up the custom region if it's too long.
            std::vector<Window> new_windows =
                    create_windows(seq_id, region_start, region_end, seq_length, bam_chunk_len,
                                   window_overlap, -1);
            windows.reserve(std::size(windows) + std::size(new_windows));
            windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
        }

        return windows;
    }
}

}  // namespace dorado::polisher
