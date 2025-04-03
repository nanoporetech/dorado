#include "polish_impl.h"

#include "hts_io/FastxRandomReader.h"
#include "secondary/architectures/model_factory.h"
#include "secondary/batching.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/region.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"

#include <ATen/ATen.h>
#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado::polisher {

namespace {

std::vector<DeviceInfo> init_devices(const std::string& devices_str) {
    std::vector<DeviceInfo> devices;

    if (devices_str == "cpu") {
        torch::Device torch_device = torch::Device(devices_str);
        devices.emplace_back(DeviceInfo{devices_str, DeviceType::CPU, std::move(torch_device)});
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(devices_str, "cuda")) {
        spdlog::debug("Parsing CUDA device string.");
        const std::vector<std::string> parsed_devices =
                dorado::utils::parse_cuda_device_string(devices_str);
        if (std::empty(parsed_devices)) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
        for (const auto& val : parsed_devices) {
            torch::Device torch_device = torch::Device(val);
            devices.emplace_back(DeviceInfo{val, DeviceType::CUDA, std::move(torch_device)});
        }
    }
#endif
    else {
        throw std::runtime_error("Unsupported device: " + devices_str);
    }

    return devices;
}

}  // namespace

PolisherResources create_resources(const secondary::ModelConfig& model_config,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   const int32_t num_bam_threads,
                                   const int32_t num_inference_threads,
                                   const bool full_precision,
                                   const std::string& read_group,
                                   const std::string& tag_name,
                                   const int32_t tag_value,
                                   const std::optional<bool>& tag_keep_missing_override,
                                   const std::optional<int32_t>& min_mapq_override) {
    PolisherResources resources;

    spdlog::info("Initializing the devices.");
    resources.devices = init_devices(device_str);
    if (std::empty(resources.devices)) {
        throw std::runtime_error("Zero devices initialized! Need at least one device to run.");
    }

    // Construct the model.
    spdlog::debug("[create_resources] Loading the model.");
    const auto create_models = [&]() {
        std::vector<std::shared_ptr<secondary::ModelTorchBase>> ret;
        std::vector<c10::optional<c10::Stream>> ret_streams;

        for (int32_t device_id = 0; device_id < dorado::ssize(resources.devices); ++device_id) {
            const auto& device_info = resources.devices[device_id];

            {
                c10::optional<c10::Stream> stream;
#if DORADO_CUDA_BUILD
                if (device_info.device.is_cuda()) {
                    c10::cuda::CUDAGuard device_guard(device_info.device);
                    stream = c10::cuda::getStreamFromPool(false, device_info.device.index());
                }
#endif

                spdlog::debug("[create_resources] Creating a model from the config.");
                auto model = secondary::model_factory(model_config);

                spdlog::debug("[create_resources] About to load model to device {}: {}", device_id,
                              device_info.name);
                model->to_device(device_info.device);

                // Half-precision if needed.
                if ((device_info.type == DeviceType::CUDA) && !full_precision) {
                    spdlog::debug("[create_resources] Converting the model to half.");
                    model->to_half();
                } else {
                    spdlog::debug("[create_resources] Using full precision.");
                }

                spdlog::debug("[create_resources] Switching model to eval mode.");
                model->set_eval();

                ret.emplace_back(std::move(model));
                ret_streams.emplace_back(std::move(stream));

                spdlog::info("Loaded model to device {}: {}", device_id, device_info.name);
            }

            const int32_t last_model = static_cast<int32_t>(std::size(ret)) - 1;
            for (int32_t i = 1; i < num_inference_threads; ++i) {
                ret.emplace_back(ret[last_model]);
                c10::optional<c10::Stream> stream;
#if DORADO_CUDA_BUILD
                if (device_info.device.is_cuda()) {
                    c10::cuda::CUDAGuard device_guard(device_info.device);
                    stream = c10::cuda::getStreamFromPool(false, device_info.device.index());
                }
#endif
                ret_streams.emplace_back(std::move(stream));
                spdlog::info("Loaded model to device {}: {}", device_id, device_info.name);
            }
        }

        return std::make_pair(std::move(ret), std::move(ret_streams));
    };
    std::tie(resources.models, resources.streams) = create_models();

    spdlog::info("Creating the encoder.");
    resources.encoder = encoder_factory(model_config, read_group, tag_name, tag_value,
                                        tag_keep_missing_override, min_mapq_override);

    spdlog::info("Creating the decoder.");
    resources.decoder = decoder_factory(model_config);

    // Open the BAM file for each thread.
    spdlog::info("Creating {} BAM handles.", num_bam_threads);
    for (int32_t i = 0; i < num_bam_threads; ++i) {
        resources.bam_handles.emplace_back(secondary::BamFile(in_aln_bam_fn));
    }

    return resources;
}

void remove_deletions(secondary::ConsensusResult& cons) {
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

std::vector<secondary::ConsensusResult> stitch_sequence(
        const hts_io::FastxRandomReader& fastx_reader,
        const std::string& header,
        const std::vector<secondary::ConsensusResult>& sample_results,
        const std::vector<std::pair<int64_t, int32_t>>& samples_for_seq,
        const bool fill_gaps,
        const std::optional<char>& fill_char) {
    const std::string draft = fastx_reader.fetch_seq(header);
    const int64_t draft_len = dorado::ssize(draft);

    if (fill_gaps && std::empty(samples_for_seq)) {
        spdlog::debug(
                "Sequence '{}' of length {} has zero inferred samples. Copying contig verbatim "
                "from input.",
                header, std::size(draft));
        std::string dummy_quals(std::size(draft), '!');
        return {secondary::ConsensusResult{header, draft, std::move(dummy_quals)}};
    } else if (!fill_gaps && std::empty(samples_for_seq)) {
        spdlog::debug(
                "Sequence '{}' of length {} has zero inferred samples. NOT copying contig "
                "verbatim from input because fill_gaps == false.",
                header, std::size(draft));
        return {};
    }

    std::vector<secondary::ConsensusResult> ret;

    secondary::ConsensusResult result;
    result.name = header;
    result.draft_start = draft_len;
    result.draft_end = 0;

    // This is an inclusive coordinate.
    int64_t last_end = 0;
    for (size_t i = 0; i < std::size(samples_for_seq); ++i) {
        const int32_t sample_index = samples_for_seq[i].second;
        const secondary::ConsensusResult& sample_result = sample_results[sample_index];

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
                result.name = header;
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

namespace {

/**
 * \brief If the input sample coordinates (positions_major) have gaps,
 *          this function splits the sample on those gaps and produces
 *          one or more samples in the output.
 *          When possible, input data is moved to the output, and that is
 *          why the inpunt is not const.
 */
std::vector<secondary::Sample> split_sample_on_discontinuities(secondary::Sample& sample) {
    std::vector<secondary::Sample> results;

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

            results.emplace_back(secondary::Sample{
                    sample.seq_id, sample.features.slice(0, start, end), std::move(new_major_pos),
                    std::move(new_minor_pos), sample.depth.slice(0, start, end),
                    std::move(read_ids_left), placeholder_ids});
            start = end;
        }

        if (start < num_positions) {
            std::vector<int64_t> new_major_pos(std::begin(sample.positions_major) + start,
                                               std::end(sample.positions_major));
            std::vector<int64_t> new_minor_pos(std::begin(sample.positions_minor) + start,
                                               std::end(sample.positions_minor));
            results.emplace_back(secondary::Sample{
                    sample.seq_id, sample.features.slice(0, start), std::move(new_major_pos),
                    std::move(new_minor_pos), sample.depth.slice(0, start), placeholder_ids,
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
std::vector<secondary::Sample> split_samples(std::vector<secondary::Sample> samples,
                                             const int64_t chunk_len,
                                             const int64_t chunk_overlap) {
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    const auto create_chunk = [](const secondary::Sample& sample, const int64_t start,
                                 const int64_t end) {
        torch::Tensor new_features = sample.features.slice(0, start, end);
        std::vector<int64_t> new_major(std::begin(sample.positions_major) + start,
                                       std::begin(sample.positions_major) + end);
        std::vector<int64_t> new_minor(std::begin(sample.positions_minor) + start,
                                       std::begin(sample.positions_minor) + end);
        torch::Tensor new_depth = sample.depth.slice(0, start, end);
        return secondary::Sample{
                sample.seq_id,
                std::move(new_features),
                std::move(new_major),
                std::move(new_minor),
                std::move(new_depth),
                {},
                {},
        };
    };

    std::vector<secondary::Sample> results;
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

}  // namespace

std::pair<std::vector<secondary::Sample>, std::vector<secondary::TrimInfo>>
merge_and_split_bam_regions_in_parallel(std::vector<secondary::Sample>& window_samples,
                                        const secondary::EncoderBase& encoder,
                                        const Span<const secondary::Window> bam_regions,
                                        const Span<const secondary::Interval> bam_region_intervals,
                                        const int32_t num_threads,
                                        const int32_t window_len,
                                        const int32_t window_overlap,
                                        const int32_t window_interval_offset) {
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
    const auto debug_print_samples =
            [](std::ostream& os, const std::vector<secondary::Sample>& samples,
               int64_t start /* = 0*/, int64_t end /* = -1 */, int64_t debug_id /* = -1 */) {
                start = std::max<int64_t>(0, start);
                end = (end <= 0) ? static_cast<int64_t>(std::size(samples)) : end;

                for (int64_t i = start; i < end; ++i) {
                    os << "[i = " << i << "] ";
                    debug_print_sample(os, samples[i], 0, -1, i == debug_id);
                    os << '\n';
                }
            };
#endif

    utils::ScopedProfileRange spr1("merge_and_split_bam_regions_in_parallel", 3);

    const auto worker = [&](const int32_t start, const int32_t end,
                            std::vector<std::vector<secondary::Sample>>& results_samples,
                            std::vector<std::vector<secondary::TrimInfo>>& results_trims) {
        utils::ScopedProfileRange spr2("merge_and_split_bam_regions_in_parallel-worker", 4);

        for (int32_t bam_region_id = start; bam_region_id < end; ++bam_region_id) {
            // Get the interval of samples for this BAM region and subtract the offset due to batching.
            secondary::Interval interval = bam_region_intervals[bam_region_id];
            interval.start -= window_interval_offset;
            interval.end -= window_interval_offset;

            spdlog::trace("- [bam_region_id = {}] (0) Before merging: interval = [{}, {}]",
                          bam_region_id, interval.start, interval.end);
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            debug_print_samples(std::cerr, window_samples, interval.start, interval.end, -1);
#endif

            std::vector<secondary::Sample> local_samples;

            // Split all samples on discontinuities.
            for (int32_t sample_id = interval.start; sample_id < interval.end; ++sample_id) {
                auto& sample = window_samples[sample_id];
                std::vector<secondary::Sample> disc_samples =
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
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            // Merge adjacent samples.
            local_samples = encoder.merge_adjacent_samples(local_samples);

            spdlog::trace("- [bam_region_id = {}] (2) After merging adjacent: local_samples = {}",
                          bam_region_id, std::size(local_samples));
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            // Bluntly split samples for inference.
            local_samples = split_samples(std::move(local_samples), window_len, window_overlap);

            spdlog::trace(
                    "- [bam_region_id = {}] (3) After splitting samples: local_samples = {}, "
                    "window_len = {}, window_overlap = {}",
                    bam_region_id, std::size(local_samples), window_len, window_overlap);
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
            debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

            // Compute sample trimming coordinates.
            const secondary::Window& reg = bam_regions[bam_region_id];
            results_trims[bam_region_id] = trim_samples(
                    local_samples, std::optional<secondary::RegionInt>(
                                           {reg.seq_id, reg.start_no_overlap, reg.end_no_overlap}));

            // Filter local_samples on non-valid trims before returning.
            std::vector<secondary::Sample> filt_local_samples;
            std::vector<secondary::TrimInfo> filt_trims;
            filt_local_samples.reserve(std::size(local_samples));
            filt_trims.reserve(std::size(local_samples));
            assert(std::size(local_samples) == std::size(results_trims[bam_region_id]));
            for (size_t sample_id = 0; sample_id < std::size(local_samples); ++sample_id) {
                auto& trim = results_trims[bam_region_id][sample_id];
                if (!secondary::is_trim_info_valid(trim)) {
                    continue;
                }
                filt_local_samples.emplace_back(std::move(local_samples[sample_id]));
                filt_trims.emplace_back(std::move(trim));
            }
            results_samples[bam_region_id] = std::move(filt_local_samples);
            results_trims[bam_region_id] = std::move(filt_trims);
        }
    };

    // Result vectors for each BAM region.
    std::vector<std::vector<secondary::Sample>> merged_samples(std::size(bam_region_intervals));
    std::vector<std::vector<secondary::TrimInfo>> merged_trims(std::size(bam_region_intervals));

    // Process BAM regions in parallel.
    const std::vector<secondary::Interval> thread_chunks = secondary::compute_partitions(
            static_cast<int32_t>(std::size(bam_region_intervals)), num_threads);

    spdlog::trace("Starting to merge samples for {} BAM windows using {} threads.",
                  std::size(bam_region_intervals), std::size(thread_chunks));

    // Parallel processing of BAM regions.
    cxxpool::thread_pool pool{std::size(thread_chunks)};
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));
    for (size_t tid = 0; tid < std::size(thread_chunks); ++tid) {
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

    // Flatten the samples obtained for each BAM region.
    size_t num_samples = 0;
    for (const auto& vals : merged_samples) {
        num_samples += std::size(vals);
    }

    std::vector<secondary::Sample> results_samples;
    results_samples.reserve(num_samples);
    for (auto& vals : merged_samples) {
        results_samples.insert(std::end(results_samples), std::make_move_iterator(std::begin(vals)),
                               std::make_move_iterator(std::end(vals)));
    }

    std::vector<secondary::TrimInfo> results_trims;
    results_trims.reserve(num_samples);
    for (auto& vals : merged_trims) {
        results_trims.insert(std::end(results_trims), std::make_move_iterator(std::begin(vals)),
                             std::make_move_iterator(std::end(vals)));
    }

    return {results_samples, results_trims};
}

std::vector<secondary::Sample> encode_windows_in_parallel(
        std::vector<secondary::BamFile>& bam_handles,
        const secondary::EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const dorado::Span<const secondary::Window> windows,
        const int32_t num_threads) {
    utils::ScopedProfileRange spr1("encode_windows_in_parallel", 3);

    // Worker function, each thread computes tensors for a set of windows assigned to it.
    const auto worker = [&](const int32_t thread_id, const int32_t start, const int32_t end,
                            std::vector<secondary::Sample>& results) {
        utils::ScopedProfileRange spr2("encode_windows_in_parallel-worker", 4);

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

    const std::vector<secondary::Interval> thread_chunks = secondary::compute_partitions(
            static_cast<int32_t>(std::size(windows)),
            std::min(num_threads, static_cast<int32_t>(std::size(bam_handles))));

    spdlog::debug("Starting to encode regions for {} windows using {} threads.", std::size(windows),
                  std::size(thread_chunks));

    // Create the thread pool, futures and results.
    cxxpool::thread_pool pool{std::size(thread_chunks)};
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));

    std::vector<secondary::Sample> results(std::size(windows));

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

std::vector<secondary::Window> create_windows_from_regions(
        const std::vector<secondary::Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap) {
    utils::ScopedProfileRange spr1("create_windows_from_regions", 2);

    std::vector<secondary::Window> windows;

    for (auto region : regions) {
        spdlog::debug("Creating windows for region: '{}'.", region_to_string(region));

        const auto it = draft_lookup.find(region.name);
        if (it == std::end(draft_lookup)) {
            throw std::runtime_error(
                    "Sequence specified by custom region not found in input! Sequence name: " +
                    region.name);
        }
        const auto [seq_id, seq_length] = it->second;

        region.start = std::max<int64_t>(0, region.start);
        region.end = (region.end < 0) ? seq_length : std::min(seq_length, region.end);

        if (region.start >= region.end) {
            throw std::runtime_error{"Region coordinates not valid. Given: region.name = '" +
                                     region.name +
                                     "', region.start = " + std::to_string(region.start) +
                                     ", region.end = " + std::to_string(region.end)};
        }

        // Split the custom region if it's too long.
        std::vector<secondary::Window> new_windows =
                secondary::create_windows(static_cast<int32_t>(seq_id), region.start, region.end,
                                          seq_length, bam_chunk_len, window_overlap);

        spdlog::debug("Generated {} windows for region: '{}'.", std::size(new_windows),
                      region_to_string(region));
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    return windows;
}

void sample_producer(PolisherResources& resources,
                     const std::vector<secondary::Window>& bam_regions,
                     const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                     const int32_t num_threads,
                     const int32_t batch_size,
                     const int32_t window_len,
                     const int32_t window_overlap,
                     const int32_t bam_subchunk_len,
                     utils::AsyncQueue<InferenceData>& infer_data) {
    utils::ScopedProfileRange spr1("sample_producer", 2);

    spdlog::debug("[producer] Input: {} BAM windows.", std::size(bam_regions));

    // Split large BAM regions into non-overlapping windows for parallel encoding.
    // The non-overlapping windows will be merged after samples are constructed.
    std::vector<secondary::Window> windows;
    std::vector<secondary::Interval> bam_region_intervals;
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(bam_regions)); ++i) {
        const secondary::Window& bw = bam_regions[i];
        std::vector<secondary::Window> new_windows = secondary::create_windows(
                bw.seq_id, bw.start, bw.end, bw.seq_length, bam_subchunk_len, 0);
        if (std::empty(new_windows)) {
            bam_region_intervals.emplace_back(secondary::Interval{0, 0});
            continue;
        }
        const int32_t num_windows = static_cast<int32_t>(std::size(windows));
        const int32_t num_new_windows = static_cast<int32_t>(std::size(new_windows));
        bam_region_intervals.emplace_back(
                secondary::Interval{num_windows, num_windows + num_new_windows});
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    const std::vector<secondary::Interval> bam_region_batches = secondary::create_batches(
            bam_region_intervals, num_threads,
            [](const secondary::Interval& val) { return val.end - val.start; });

    InferenceData buffer;

    // Each iteration of the for loop produces full BAM regions of samples to fit at least num_threads windows.
    // It is important to process full BAM regions because of splitting/merging/splitting and trimming.
    for (const auto [region_id_start, region_id_end] : bam_region_batches) {
        if (region_id_end <= region_id_start) {
            continue;
        }

        try {
            const int32_t num_regions = region_id_end - region_id_start;
            const int32_t window_id_start = bam_region_intervals[region_id_start].start;
            const int32_t window_id_end = bam_region_intervals[region_id_end - 1].end;
            const size_t num_windows = static_cast<size_t>(window_id_end - window_id_start);

            // Encode samples in parallel. Non-const by design, data will be moved.
            std::vector<secondary::Sample> region_samples = encode_windows_in_parallel(
                    resources.bam_handles, *resources.encoder, draft_lens,
                    Span<const secondary::Window>(std::data(windows) + window_id_start,
                                                  num_windows),
                    num_threads);

            spdlog::trace(
                    "[producer] Merging the samples into {} BAM chunks. parallel_results.size() = "
                    "{}",
                    num_regions, std::size(region_samples));

            auto [samples, trims] = merge_and_split_bam_regions_in_parallel(
                    region_samples, *resources.encoder,
                    Span<const secondary::Window>(std::data(bam_regions) + region_id_start,
                                                  num_regions),
                    Span<const secondary::Interval>(
                            std::data(bam_region_intervals) + region_id_start, num_regions),
                    num_threads, window_len, window_overlap, window_id_start);

            if (std::size(samples) != std::size(trims)) {
                throw std::runtime_error(
                        "Size of samples and trims does not match! samples.size() = " +
                        std::to_string(std::size(samples)) +
                        ", trims.size() = " + std::to_string(std::size(trims)));
            }

            // Add samples to the batches.
            for (size_t i = 0; i < std::size(samples); ++i) {
                // If any of the samples is of wrong size, create a remainder batch of 1.
                if (dorado::ssize(samples[i].positions_major) != window_len) {
                    InferenceData remainder_buffer;
                    remainder_buffer.samples = {std::move(samples[i])};
                    remainder_buffer.trims = {std::move(trims[i])};
                    spdlog::trace(
                            "[producer] Pushing a batch of data to infer_data queue. "
                            "remainder_buffer.samples.size() = {}",
                            std::size(remainder_buffer.samples));
                    infer_data.try_push(std::move(remainder_buffer));
                    continue;
                }

                // Expand the current buffer.
                buffer.samples.emplace_back(std::move(samples[i]));
                buffer.trims.emplace_back(std::move(trims[i]));

                if (dorado::ssize(buffer.samples) == batch_size) {
                    spdlog::trace(
                            "[producer] Pushing a batch of data to infer_data queue. "
                            "buffer.samples.size() = {}",
                            std::size(buffer.samples));
                    infer_data.try_push(std::move(buffer));
                    buffer = {};
                }
            }
        } catch (const std::exception& e) {
            spdlog::warn(
                    "Caught exception in the sample producer. Skipping the remaining regions in "
                    "current batch (region_id_start = {}, region_id_end = {}). Exception: {}",
                    region_id_start, region_id_end, e.what());
            continue;
        }
    }

    if (!std::empty(buffer.samples)) {
        spdlog::trace(
                "[producer] Pushing a batch of data to infer_data queue. "
                "buffer.samples.size() = {} (final)",
                std::size(buffer.samples));
        infer_data.try_push(std::move(buffer));
        buffer = {};
        spdlog::debug("[producer] Pushed final batch for inference to infer_data queue.");
    }

    infer_data.terminate();
}

void infer_samples_in_parallel(utils::AsyncQueue<InferenceData>& batch_queue,
                               utils::AsyncQueue<DecodeData>& decode_queue,
                               std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
                               const std::vector<c10::optional<c10::Stream>>& streams,
                               const secondary::EncoderBase& encoder) {
    utils::ScopedProfileRange spr1("infer_samples_in_parallel", 2);

    if (std::empty(models)) {
        throw std::runtime_error("No models have been initialized, cannot run inference.");
    }

    auto batch_infer = [&encoder](secondary::ModelTorchBase& model, const InferenceData& batch,
                                  const int32_t tid) {
        utils::ScopedProfileRange spr2("infer_samples_in_parallel-batch_infer", 3);
        timer::TimerHighRes timer_total;

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        timer::TimerHighRes timer_collate;
        torch::Tensor batch_features_tensor;
        int64_t time_collate = 0;
        {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-collate", 4);
            std::vector<torch::Tensor> batch_features;
            batch_features.reserve(std::size(batch.samples));
            for (const auto& sample : batch.samples) {
                batch_features.emplace_back(sample.features);
            }
            batch_features_tensor = encoder.collate(std::move(batch_features));
            time_collate = timer_collate.GetElapsedMilliseconds();
        }

        // Debug output.
        {
            spdlog::trace(
                    "[consumer {}] About to call forward(): batch_features_tensor.size() = [{}], "
                    "approx "
                    "size: {} MB.",
                    tid, utils::tensor_shape_as_string(batch_features_tensor),
                    batch_features_tensor.numel() * batch_features_tensor.element_size() /
                            (1024.0 * 1024.0));
        }

        // Inference.
        torch::Tensor output;
        timer::TimerHighRes timer_forward;

        {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-infer", 4);

            std::unique_lock<std::mutex> lock;

#if DORADO_CUDA_BUILD
            if (model.get_device() == torch::kCUDA) {
                lock = dorado::utils::acquire_gpu_lock(model.get_device().index(), true);
            }
#endif

            try {
                output = model.predict_on_batch(std::move(batch_features_tensor));
            } catch (std::exception& e) {
                spdlog::error("ERROR! Exception caught: {}", e.what());
                throw;
            }
        }

        // Debug output.
        {
            const int64_t time_forward = timer_forward.GetElapsedMilliseconds();
            const int64_t time_total = timer_total.GetElapsedMilliseconds();

            spdlog::trace(
                    "[consumer {}] Computed batch inference. Timings - collate: {} ms, forward: {} "
                    "ms, "
                    "total = {}",
                    tid, time_collate, time_forward, time_total);
        }

        return output;
    };

    const auto worker = [&](const int32_t tid, secondary::ModelTorchBase& model,
                            [[maybe_unused]] const c10::optional<c10::Stream>& stream) {
        utils::ScopedProfileRange spr2("infer_samples_in_parallel-worker", 3);

#if DORADO_CUDA_BUILD
        c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

        at::InferenceMode infer_guard;

        while (true) {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-worker-while", 4);

            spdlog::trace("[consumer {}] Waiting to pop data for inference. Queue size: {}", tid,
                          std::size(batch_queue));

            InferenceData item;
            const auto pop_status = batch_queue.try_pop(item);

            spdlog::trace("[consumer {}] Popped data: item.samples.size() = {}, queue size: {}",
                          tid, std::size(item.samples), batch_queue.size());

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            if (std::empty(item.samples)) {
                continue;
            }

            // Inference.
            try {
                torch::Tensor logits = batch_infer(model, item, tid);

                // One out_item contains samples for one inference batch.
                // No guarantees on any sort of logical ordering of the samples.
                DecodeData out_item;
                out_item.samples = std::move(item.samples);
                out_item.logits = std::move(logits);
                out_item.trims = std::move(item.trims);

                spdlog::trace(
                        "[consumer {}] Pushing data to decode_queue: out_item.logits.shape = {} "
                        "out_item.samples.size() = {}, decode queue size: {}",
                        tid, utils::tensor_shape_as_string(out_item.logits),
                        std::size(out_item.samples), std::size(decode_queue));
                decode_queue.try_push(std::move(out_item));

            } catch (const std::exception& e) {
                spdlog::warn(
                        "Caught exception when inferring a batch of samples. Skipping this "
                        "batch. Exception: {}",
                        e.what());
            }
        }
    };

    const size_t num_threads = std::size(models);
    cxxpool::thread_pool pool{num_threads};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(*models[tid]), streams[tid]));
    }

    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from inference task: ") + e.what()};
    }

    decode_queue.terminate();

    spdlog::debug("[infer_samples_in_parallel] Finished running inference.");
}

void decode_samples_in_parallel(std::vector<secondary::ConsensusResult>& results_cons,
                                std::vector<secondary::VariantCallingSample>& results_vc_data,
                                utils::AsyncQueue<DecodeData>& decode_queue,
                                PolishStats& polish_stats,
                                const secondary::DecoderBase& decoder,
                                const int32_t num_threads,
                                const int32_t min_depth,
                                const bool collect_vc_data) {
    utils::ScopedProfileRange spr1("decode_samples_in_parallel", 2);

    auto batch_decode = [&decoder, &polish_stats, min_depth](const DecodeData& item,
                                                             const int32_t tid) {
        utils::ScopedProfileRange spr2("decode_samples_in_parallel-batch_decode", 3);

        timer::TimerHighRes timer_total;
        timer::TimerHighRes timer_decode;

        // Decode output to bases and qualities.
        std::vector<secondary::ConsensusResult> local_results = decoder.decode_bases(item.logits);

        std::vector<secondary::ConsensusResult> final_results;
        final_results.reserve(std::size(local_results));

        const int64_t time_decode = timer_decode.GetElapsedMilliseconds();

        assert(std::size(local_results) == std::size(item.samples));
        assert(std::size(local_results) == std::size(item.trims));

        // Trim the overlapping sequences.
        timer::TimerHighRes timer_trim;
        for (int64_t j = 0; j < dorado::ssize(local_results); ++j) {
            auto& result = local_results[j];
            const secondary::Sample& sample = item.samples[j];
            const secondary::TrimInfo& trim = item.trims[j];
            const int64_t num_positions = dorado::ssize(sample.positions_major);

            if ((trim.start < 0) || (trim.start >= num_positions) || (trim.end <= 0) ||
                (trim.end > num_positions)) {
                spdlog::debug(
                        "Trim coordinate is < 0. j = {}, trim.start = {}, trim.end = {}, "
                        "trim.heuristic = {}, num_positions = {}",
                        j, trim.start, trim.end, trim.heuristic, num_positions);
                result = {};
                continue;
            }

            std::vector<secondary::Interval> good_intervals{
                    secondary::Interval{0, static_cast<int32_t>(num_positions)}};

            // Break on low coverage.
            if (min_depth > 0) {
                good_intervals.clear();

                const Span<int64_t> depth(sample.depth.data_ptr<int64_t>(),
                                          static_cast<size_t>(sample.depth.size(0)));
                secondary::Interval interval{0, 0};
                for (int32_t ii = 0; ii < static_cast<int32_t>(std::size(depth)); ++ii) {
                    if (depth[ii] < min_depth) {
                        if (interval.length() > 0) {
                            good_intervals.emplace_back(interval);
                        }
                        interval.start = ii + 1;
                    }
                    interval.end = ii + 1;
                }
                if (interval.length() > 0) {
                    good_intervals.emplace_back(interval);
                }
            }

            if (std::size(good_intervals) == 1) {
                // Trim and mark the region.
                result.draft_id = sample.seq_id;
                result.draft_start = sample.positions_major[trim.start];
                result.draft_end = sample.positions_major[trim.end - 1] + 1;
                result.seq = result.seq.substr(trim.start, trim.end - trim.start);
                result.quals = result.quals.substr(trim.start, trim.end - trim.start);

                final_results.emplace_back(std::move(result));

            } else {
                for (const auto& interval : good_intervals) {
                    if ((interval.start < 0) || (interval.end <= 0)) {
                        continue;
                    }

                    secondary::ConsensusResult new_result;

                    const int32_t start =
                            std::max(static_cast<int32_t>(trim.start), interval.start);
                    const int32_t end = std::min(static_cast<int32_t>(trim.end), interval.end);

                    if (end <= start) {
                        continue;
                    }

                    new_result.draft_id = sample.seq_id;
                    new_result.draft_start = sample.positions_major[start];
                    new_result.draft_end = sample.positions_major[end - 1] + 1;
                    new_result.seq = result.seq.substr(start, end - start);
                    new_result.quals = result.quals.substr(start, end - start);

                    final_results.emplace_back(std::move(new_result));
                }
            }

            polish_stats.add("processed",
                             static_cast<double>(result.draft_end - result.draft_start));
        }
        const int64_t time_trim = timer_trim.GetElapsedMilliseconds();

        const int64_t time_total = timer_total.GetElapsedMilliseconds();

        spdlog::trace(
                "[decoder {}] Computed batch decode. Timings - decode = {} "
                "ms, trim = {} ms, total = {}",
                tid, time_decode, time_trim, time_total);

        return final_results;
    };

    const auto worker = [&](const int32_t tid,
                            std::vector<secondary::ConsensusResult>& thread_results,
                            std::vector<secondary::VariantCallingSample>& thread_vc_data) {
        utils::ScopedProfileRange spr2("decode_samples_in_parallel-worker", 3);
        at::InferenceMode infer_guard;

        while (true) {
            utils::ScopedProfileRange spr3("decode_samples_in_parallel-worker-while", 4);

            DecodeData item;
            const auto pop_status = decode_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            try {
                const int64_t tensor_batch_size =
                        (item.logits.sizes().size() == 0) ? 0 : item.logits.size(0);

                assert(tensor_batch_size == dorado::ssize(item.trims));

                spdlog::trace(
                        "[decoder {}] Popped data: item.logits.shape = {}, item.trims.size = {}, "
                        "tensor_batch_size = {}, queue size: {}",
                        tid, utils::tensor_shape_as_string(item.logits), dorado::ssize(item.trims),
                        tensor_batch_size, std::size(decode_queue));

                // This should handle the timeout case too.
                if (tensor_batch_size == 0) {
                    continue;
                }

                // Inference.
                std::vector<secondary::ConsensusResult> results_samples = batch_decode(item, tid);

                thread_results.insert(std::end(thread_results),
                                      std::make_move_iterator(std::begin(results_samples)),
                                      std::make_move_iterator(std::end(results_samples)));

                // Separate the logits for each sample.
                if (collect_vc_data) {
                    // Split logits for each input sample in the batch, and check that the number matches.
                    const std::vector<torch::Tensor> split_logits = item.logits.unbind(0);
                    if (std::size(item.samples) != std::size(split_logits)) {
                        spdlog::error(
                                "The number of logits produced by the batch inference does not "
                                "match "
                                "the "
                                "input batch size! Variant calling for this batch will not be "
                                "possible. "
                                "samples.size = {}, split_logits.size = {}",
                                std::size(item.samples), std::size(split_logits));
                        continue;
                    }
                    // Create the variant calling data. Clone the tensor to convert the view to actual data.
                    for (int64_t i = 0; i < dorado::ssize(item.samples); ++i) {
                        thread_vc_data.emplace_back(secondary::VariantCallingSample{
                                item.samples[i].seq_id, std::move(item.samples[i].positions_major),
                                std::move(item.samples[i].positions_minor),
                                split_logits[i].clone()});
                    }
                }

            } catch (const std::exception& e) {
                spdlog::warn(
                        "Caught an exception when decoding a batch of samples. Skipping this "
                        "batch. Exception: {}",
                        e.what());
            }
        }
    };

    std::vector<std::vector<secondary::ConsensusResult>> thread_results(num_threads);
    std::vector<std::vector<secondary::VariantCallingSample>> thread_vc_data(num_threads);

    cxxpool::thread_pool pool{static_cast<size_t>(num_threads)};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(thread_results[tid]),
                                       std::ref(thread_vc_data[tid])));
    }

    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from inference task: ") + e.what()};
    }

    // Flatten the results.
    {
        size_t total_size = 0;
        for (const auto& vals : thread_results) {
            total_size += std::size(vals);
        }
        results_cons.clear();
        results_cons.reserve(total_size);
        for (auto& vals : thread_results) {
            results_cons.insert(std::end(results_cons), std::make_move_iterator(std::begin(vals)),
                                std::make_move_iterator(std::end(vals)));
        }
    }
    {
        size_t total_size = 0;
        for (const auto& vals : thread_vc_data) {
            total_size += std::size(vals);
        }
        results_vc_data.clear();
        results_vc_data.reserve(total_size);
        for (auto& vals : thread_vc_data) {
            results_vc_data.insert(std::end(results_vc_data),
                                   std::make_move_iterator(std::begin(vals)),
                                   std::make_move_iterator(std::end(vals)));
        }
    }

    spdlog::debug("[decode_samples_in_parallel] Finished decoding the output.");
}

std::vector<std::vector<secondary::ConsensusResult>> construct_consensus_seqs(
        const secondary::Interval& region_batch,
        const std::vector<secondary::ConsensusResult>& all_results_cons,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool fill_gaps,
        const std::optional<char>& fill_char,
        hts_io::FastxRandomReader& draft_reader) {
    utils::ScopedProfileRange spr1("construct_consensus_seqs", 3);

    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(all_results_cons); ++i) {
        const secondary::ConsensusResult& r = all_results_cons[i];
        const int32_t local_id = r.draft_id - region_batch.start;
        // Skip filtered samples.
        if (r.draft_id < 0) {
            continue;
        }
        if ((r.draft_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    r.draft_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(r.draft_start, i);
    }

    std::vector<std::vector<secondary::ConsensusResult>> ret;

    // Consensus sequence - stitch the windows and write output.
    for (int64_t group_id = 0; group_id < dorado::ssize(groups); ++group_id) {
        const int64_t seq_id = group_id + region_batch.start;

        auto& group = groups[group_id];
        std::sort(std::begin(group), std::end(group));  // Sort by start pos.

        const std::string& header = draft_lens[seq_id].first;

        std::vector<secondary::ConsensusResult> consensus = stitch_sequence(
                draft_reader, header, all_results_cons, group, fill_gaps, fill_char);

        ret.emplace_back(std::move(consensus));
    }

    return ret;
}

std::vector<secondary::Variant> call_variants(
        const secondary::Interval& region_batch,
        const std::vector<secondary::VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        polisher::PolishStats& polish_stats) {
    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(vc_input_data); ++i) {
        const auto& vc_sample = vc_input_data[i];

        const int32_t local_id = vc_sample.seq_id - region_batch.start;

        // Skip filtered samples.
        if (vc_sample.seq_id < 0) {
            continue;
        }

        if ((vc_sample.seq_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    vc_sample.seq_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(vc_sample.start(), i);
    }

    // Worker for parallel processing.
    const auto worker = [&](const int32_t tid, const int32_t start, const int32_t end,
                            std::vector<std::vector<secondary::Variant>>& results,
                            polisher::PolishStats& ps) {
        if ((start < 0) || (start >= end) || (end > dorado::ssize(results))) {
            throw std::runtime_error("Worker group_id is out of bounds! start = " +
                                     std::to_string(start) + ", end = " + std::to_string(end) +
                                     ", results.size = " + std::to_string(std::size(results)));
        }

        for (int32_t group_id = start; group_id < end; ++group_id) {
            const int64_t seq_id = group_id + region_batch.start;
            const std::string& header = draft_lens[seq_id].first;

            // Catch exceptions here to skip variant calling only on one sequence instead
            // of the entire batch.
            try {
                // Sort the group by start positions.
                auto& group = groups[group_id];
                std::stable_sort(std::begin(group), std::end(group));

                if (std::empty(group)) {
                    continue;
                }

                // Get the draft sequence.
                const std::string draft = draft_readers[tid]->fetch_seq(header);

                // Trim the overlapping portions between samples.
                const auto trimmed_vc_samples = secondary::trim_vc_samples(vc_input_data, group);

                // Break and merge samples on non-variant positions.
                // const auto joined_samples = join_samples(vc_input_data, group, trims, draft, decoder);
                const auto joined_samples = join_samples(trimmed_vc_samples, draft, decoder);

                for (const auto& vc_sample : joined_samples) {
                    std::vector<secondary::Variant> variants =
                            secondary::decode_variants(decoder, vc_sample, draft, ambig_ref, gvcf);

                    ps.add("processed", static_cast<double>(vc_sample.end() - vc_sample.start()));

                    results[group_id].insert(std::end(results[group_id]),
                                             std::make_move_iterator(std::begin(variants)),
                                             std::make_move_iterator(std::end(variants)));
                }
            } catch (const std::exception& e) {
                spdlog::warn(
                        "Caught an exception in the call_variants::worker (tid = {}, group_id = "
                        "{}, seq_id = {}, header = '{}'). Not returning any variants for this "
                        "group. Original exception: '{}'",
                        tid, group_id, seq_id, header, e.what());
                results[group_id].clear();
            }
        }
    };

    // Partition groups to chunks for multithreaded processing.
    const std::vector<secondary::Interval> thread_chunks =
            secondary::compute_partitions(static_cast<int32_t>(std::size(groups)), num_threads);

    // Create the thread pool.
    cxxpool::thread_pool pool{std::size(thread_chunks)};

    // Create the futures.
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));

    // Reserve the space for results for each individual group.
    std::vector<std::vector<secondary::Variant>> thread_results(std::size(groups));

    // Add worker tasks.
    for (int32_t tid = 0; tid < static_cast<int32_t>(std::size(thread_chunks)); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, tid, chunk_start, chunk_end,
                                       std::ref(thread_results), std::ref(polish_stats)));
    }

    // Join and catch exceptions.
    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception when computing variants: ") +
                                 e.what()};
    }

    // Flatten the results.
    std::vector<secondary::Variant> all_results;
    {
        size_t count = 0;
        for (const auto& vals : thread_results) {
            count += std::size(vals);
        }
        all_results.reserve(count);
        for (auto& vals : thread_results) {
            all_results.insert(std::end(all_results), std::make_move_iterator(std::begin(vals)),
                               std::make_move_iterator(std::end(vals)));
        }
    }

    return all_results;
}

}  // namespace dorado::polisher
