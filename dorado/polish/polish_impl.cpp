#include "polish_impl.h"

#include "polish/interval.h"
#include "polish/polish_utils.h"
#include "polish/region.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"

#include <ATen/ATen.h>
#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado::polisher {

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

PolisherResources create_resources(const ModelConfig& model_config,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   const int32_t num_bam_threads,
                                   const int32_t num_inference_cpu_threads,
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
        std::vector<std::shared_ptr<ModelTorchBase>> ret;

        for (int32_t device_id = 0; device_id < dorado::ssize(resources.devices); ++device_id) {
            const auto& device_info = resources.devices[device_id];

            spdlog::debug("[create_resources] Creating a model from the config.");
            auto model = model_factory(model_config);

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

            spdlog::info("Loaded model to device {}: {}", device_id, device_info.name);
        }
        // In case the device is set to CPU, we add up to inference_threads models.
        if ((std::size(resources.devices) == 1) &&
            (resources.devices.front().type == DeviceType::CPU)) {
            for (int32_t i = 1; i < num_inference_cpu_threads; ++i) {
                ret.emplace_back(ret.front());
            }
        }
        return ret;
    };
    resources.models = create_models();

    spdlog::info("Creating the encoder.");
    resources.encoder = encoder_factory(model_config, read_group, tag_name, tag_value,
                                        tag_keep_missing_override, min_mapq_override);

    spdlog::info("Creating the decoder.");
    resources.decoder = decoder_factory(model_config);

    // Open the BAM file for each thread.
    spdlog::info("Creating {} BAM handles.", num_bam_threads);
    for (int32_t i = 0; i < num_bam_threads; ++i) {
        resources.bam_handles.emplace_back(BamFile(in_aln_bam_fn));
    }

    return resources;
}

BamInfo analyze_bam(const std::filesystem::path& in_aln_bam_fn, const std::string& cli_read_group) {
    BamInfo ret;

    BamFile bam(in_aln_bam_fn);

    const std::vector<HeaderLineData> header = bam.parse_header();

    // Get info from headers: program and the read groups.
    for (const auto& line : header) {
        // Convert all tags into a lookup.
        const std::unordered_map<std::string, std::string> tags = [&]() {
            std::unordered_map<std::string, std::string> local_ret;
            for (const auto& [key, value] : line.tags) {
                local_ret[key] = value;
            }
            return local_ret;
        }();

        if (line.header_type == "@PG") {
            // Example PG line:
            //      @PG	ID:aligner	PP:samtools.2	PN:dorado	VN:0.0.0+2852e11d	DS:2.27-r1193

            const auto& it_pn = tags.find("PN");
            const auto& it_id = tags.find("ID");
            if ((it_pn != std::end(tags)) && it_id != std::end(tags)) {
                // Convert the program name to lowercase just in case.
                std::string pn = it_pn->second;
                std::transform(std::begin(pn), std::end(pn), std::begin(pn),
                               [](unsigned char c) { return std::tolower(c); });

                // Convert the program ID to lowercase just in case.
                std::string id = it_id->second;
                std::transform(std::begin(id), std::end(id), std::begin(id),
                               [](unsigned char c) { return std::tolower(c); });

                if ((pn == "dorado") && utils::starts_with(id, "aligner")) {
                    // Multiple tools can be run on a BAM, and the ID field needs to be unique by spec.
                    // Example possibilites: aligner, aligner.1, samtools.1, samtools.2, etc.
                    ret.uses_dorado_aligner = true;
                }
            }
        } else if (line.header_type == "@RG") {
            // Example RG line:
            //      @RG	ID:e705d8cfbbe8a6bc43a865c71ace09553e8f15cd_dna_r10.4.1_e8.2_400bps_hac@v5.0.0	DT:2022-10-18T10:38:07.247961+00:00	DS:runid=e705d8cfbbe8a6bc43a865c71ace09553e8f15cd basecall_model=dna_r10.4.1_e8.2_400bps_hac@v5.0.0 modbase_models=dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v2,dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v2	LB:PCR_zymo	PL:ONT	PM:4A	PU:PAM93185	al:PCR_zymo

            // Parse the read group ID.
            const auto& it_id = tags.find("ID");
            const std::string id = (it_id != std::end(tags)) ? it_id->second : "";

            // Parse the basecaller model.
            const auto& it_ds = tags.find("DS");
            std::string basecaller_model;
            if (it_ds != std::end(tags)) {
                const std::vector<std::string> tokens = utils::split(it_ds->second, ' ');
                constexpr std::string_view TOKEN_NAME{"basecall_model="};
                for (const auto& token : tokens) {
                    if (!utils::starts_with(token, TOKEN_NAME)) {
                        continue;
                    }
                    basecaller_model = token.substr(std::size(TOKEN_NAME));
                    break;
                }
            }

            if (std::empty(id)) {
                continue;
            }
            if (!std::empty(cli_read_group) && (id != cli_read_group)) {
                continue;
            }
            if (std::empty(basecaller_model)) {
                continue;
            }

            ret.read_groups.emplace(id);
            ret.basecaller_models.emplace(basecaller_model);
        }
    }

    // Check for the dwells ("mv") tag. Only parse one record.
    {
        const auto record = bam.get_next();
        if ((record != nullptr) && (bam_aux_get(record.get(), "mv") != nullptr)) {
            ret.has_dwells = true;
        }
    }

    return ret;
}

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

std::pair<std::vector<Sample>, std::vector<TrimInfo>> merge_and_split_bam_regions_in_parallel(
        std::vector<Sample>& window_samples,
        const EncoderBase& encoder,
        const Span<const Window> bam_regions,
        const Span<const Interval> bam_region_intervals,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t window_interval_offset) {
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
            // Get the interval of samples for this BAM region and subtract the offset due to batching.
            Interval interval = bam_region_intervals[bam_region_id];
            interval.start -= window_interval_offset;
            interval.end -= window_interval_offset;

            spdlog::trace("- [bam_region_id = {}] (0) Before merging: interval = [{}, {}]",
                          bam_region_id, interval.start, interval.end);
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
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
            const Window& reg = bam_regions[bam_region_id];
            results_trims[bam_region_id] = trim_samples(
                    local_samples, std::optional<RegionInt>(
                                           {reg.seq_id, reg.start_no_overlap, reg.end_no_overlap}));
            results_samples[bam_region_id] = std::move(local_samples);
        }
    };

    // Result vectors for each BAM region.
    std::vector<std::vector<Sample>> merged_samples(std::size(bam_region_intervals));
    std::vector<std::vector<TrimInfo>> merged_trims(std::size(bam_region_intervals));

    // Process BAM regions in parallel.
    const std::vector<Interval> thread_chunks =
            compute_partitions(static_cast<int32_t>(std::size(bam_region_intervals)), num_threads);

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

std::vector<Sample> encode_windows_in_parallel(
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

std::vector<Window> create_windows_from_regions(
        const std::vector<Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap) {
    std::vector<Window> windows;

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
        std::vector<Window> new_windows =
                create_windows(static_cast<int32_t>(seq_id), region.start, region.end, seq_length,
                               bam_chunk_len, window_overlap);

        spdlog::debug("Generated {} windows for region: '{}'.", std::size(new_windows),
                      region_to_string(region));
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    return windows;
}

void sample_producer(PolisherResources& resources,
                     const std::vector<Window>& bam_regions,
                     const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                     const int32_t num_threads,
                     const int32_t batch_size,
                     const int32_t window_len,
                     const int32_t window_overlap,
                     const int32_t bam_subchunk_len,
                     utils::AsyncQueue<InferenceData>& infer_data) {
    spdlog::debug("[producer] Input: {} BAM windows.", std::size(bam_regions));

    // Split large BAM regions into non-overlapping windows for parallel encoding.
    // The non-overlapping windows will be merged after samples are constructed.
    std::vector<Window> windows;
    std::vector<Interval> bam_region_intervals;
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(bam_regions)); ++i) {
        const Window& bw = bam_regions[i];
        std::vector<Window> new_windows =
                create_windows(bw.seq_id, bw.start, bw.end, bw.seq_length, bam_subchunk_len, 0);
        if (std::empty(new_windows)) {
            bam_region_intervals.emplace_back(Interval{0, 0});
            continue;
        }
        const int32_t num_windows = static_cast<int32_t>(std::size(windows));
        const int32_t num_new_windows = static_cast<int32_t>(std::size(new_windows));
        bam_region_intervals.emplace_back(Interval{num_windows, num_windows + num_new_windows});
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    const std::vector<Interval> bam_region_batches =
            create_batches(bam_region_intervals, num_threads,
                           [](const Interval& val) { return val.end - val.start; });

    InferenceData buffer;

    // Each iteration of the for loop produces full BAM regions of samples to fit at least num_threads windows.
    // It is important to process full BAM regions because of splitting/merging/splitting and trimming.
    for (const auto [region_id_start, region_id_end] : bam_region_batches) {
        if (region_id_end <= region_id_start) {
            continue;
        }

        const int32_t num_regions = region_id_end - region_id_start;
        const int32_t window_id_start = bam_region_intervals[region_id_start].start;
        const int32_t window_id_end = bam_region_intervals[region_id_end - 1].end;
        const size_t num_windows = static_cast<size_t>(window_id_end - window_id_start);

        // Encode samples in parallel. Non-const by design, data will be moved.
        std::vector<Sample> region_samples = encode_windows_in_parallel(
                resources.bam_handles, *resources.encoder, draft_lens,
                Span<const Window>(std::data(windows) + window_id_start, num_windows), num_threads);

        spdlog::trace(
                "[producer] Merging the samples into {} BAM chunks. parallel_results.size() = {}",
                num_regions, std::size(region_samples));

        auto [samples, trims] = merge_and_split_bam_regions_in_parallel(
                region_samples, *resources.encoder,
                Span<const Window>(std::data(bam_regions) + region_id_start, num_regions),
                Span<const Interval>(std::data(bam_region_intervals) + region_id_start,
                                     num_regions),
                num_threads, window_len, window_overlap, window_id_start);

        if (std::size(samples) != std::size(trims)) {
            throw std::runtime_error("Size of samples and trims does not match! samples.size() = " +
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
                               std::vector<std::shared_ptr<ModelTorchBase>>& models,
                               const EncoderBase& encoder) {
    if (std::empty(models)) {
        throw std::runtime_error("No models have been initialized, cannot run inference.");
    }

    auto batch_infer = [&encoder](ModelTorchBase& model, const InferenceData& batch,
                                  const int32_t tid) {
        utils::ScopedProfileRange infer("infer", 1);
        timer::TimerHighRes timer_total;

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        timer::TimerHighRes timer_collate;
        std::vector<torch::Tensor> batch_features;
        batch_features.reserve(std::size(batch.samples));
        for (const auto& sample : batch.samples) {
            batch_features.emplace_back(sample.features);
        }
        torch::Tensor batch_features_tensor = encoder.collate(std::move(batch_features));
        const int64_t time_collate = timer_collate.GetElapsedMilliseconds();

        // Debug output.
        {
            std::ostringstream oss;
            print_tensor_shape(oss, batch_features_tensor);
            spdlog::trace(
                    "[consumer {}] About to call forward(): batch_features_tensor.size() = {}, "
                    "approx "
                    "size: {} MB.",
                    tid, oss.str(),
                    batch_features_tensor.numel() * batch_features_tensor.element_size() /
                            (1024.0 * 1024.0));
        }

        // Inference.
        timer::TimerHighRes timer_forward;
        torch::Tensor output;
        try {
            output = model.predict_on_batch(std::move(batch_features_tensor));
        } catch (std::exception& e) {
            spdlog::error("ERROR! Exception caught: {}", e.what());
            throw;
        }
        const int64_t time_forward = timer_forward.GetElapsedMilliseconds();

        const int64_t time_total = timer_total.GetElapsedMilliseconds();

        spdlog::trace(
                "[consumer {}] Computed batch inference. Timings - collate: {} ms, forward: {} ms, "
                "total = {}",
                tid, time_collate, time_forward, time_total);

        return output;
    };

    const auto worker = [&](const int32_t tid) {
        at::InferenceMode infer_guard;

        while (true) {
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
            torch::Tensor logits = batch_infer(*models[tid], item, tid);

            DecodeData out_item;
            out_item.samples = std::move(item.samples);
            out_item.logits = std::move(logits);
            out_item.trims = std::move(item.trims);

            spdlog::trace(
                    "[consumer {}] Pushing data to decode_queue: out_item.logits.shape = {} "
                    "out_item.samples.size() = {}, decode queue size: {}",
                    tid, tensor_shape_as_string(out_item.logits), std::size(out_item.samples),
                    std::size(decode_queue));
            decode_queue.try_push(std::move(out_item));
        }
    };

    const size_t num_threads = std::size(models);
    cxxpool::thread_pool pool{num_threads};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid));
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

void decode_samples_in_parallel(std::vector<ConsensusResult>& results,
                                utils::AsyncQueue<DecodeData>& decode_queue,
                                PolishStats& polish_stats,
                                const DecoderBase& decoder,
                                const int32_t num_threads,
                                const int32_t min_depth) {
    auto batch_decode = [&decoder, &polish_stats, min_depth](const DecodeData& item,
                                                             const int32_t tid) {
        utils::ScopedProfileRange scope_decode("decode", 1);
        timer::TimerHighRes timer_total;

        timer::TimerHighRes timer_decode;

        // Decode output to bases and qualities.
        std::vector<ConsensusResult> local_results = decoder.decode_bases(item.logits);

        std::vector<ConsensusResult> final_results;
        final_results.reserve(std::size(local_results));

        const int64_t time_decode = timer_decode.GetElapsedMilliseconds();

        assert(std::size(local_results) == std::size(item.samples));
        assert(std::size(local_results) == std::size(item.trims));

        // Trim the overlapping sequences.
        timer::TimerHighRes timer_trim;
        for (int64_t j = 0; j < dorado::ssize(local_results); ++j) {
            auto& result = local_results[j];
            const Sample& sample = item.samples[j];
            const TrimInfo& trim = item.trims[j];
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

            std::vector<Interval> good_intervals{Interval{0, static_cast<int32_t>(num_positions)}};

            if (min_depth > 0) {
                good_intervals.clear();

                const Span<int64_t> depth(sample.depth.data_ptr<int64_t>(),
                                          static_cast<size_t>(sample.depth.size(0)));
                Interval interval{0, 0};
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

                    ConsensusResult new_result;

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

    const auto worker = [&](const int32_t tid, std::vector<ConsensusResult>& thread_results) {
        at::InferenceMode infer_guard;

        while (true) {
            DecodeData item;
            const auto pop_status = decode_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            const int64_t tensor_batch_size =
                    (item.logits.sizes().size() == 0) ? 0 : item.logits.size(0);

            assert(tensor_batch_size == dorado::ssize(item.trims));

            spdlog::trace(
                    "[decoder {}] Popped data: item.logits.shape = {}, item.trims.size = {}, "
                    "tensor_batch_size = {}, queue size: {}",
                    tid, tensor_shape_as_string(item.logits), dorado::ssize(item.trims),
                    tensor_batch_size, std::size(decode_queue));

            // This should handle the timeout case too.
            if (tensor_batch_size == 0) {
                continue;
            }

            // Inference.
            std::vector<ConsensusResult> results_samples = batch_decode(item, tid);

            thread_results.insert(std::end(thread_results),
                                  std::make_move_iterator(std::begin(results_samples)),
                                  std::make_move_iterator(std::end(results_samples)));
        }
    };

    std::vector<std::vector<ConsensusResult>> thread_results(num_threads);

    cxxpool::thread_pool pool{static_cast<size_t>(num_threads)};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(thread_results[tid])));
    }

    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from inferencetask: ") + e.what()};
    }

    // Flatten the results.
    size_t total_size = 0;
    for (const auto& vals : thread_results) {
        total_size += std::size(vals);
    }
    results.clear();
    results.reserve(total_size);
    for (auto& vals : thread_results) {
        results.insert(std::end(results), std::make_move_iterator(std::begin(vals)),
                       std::make_move_iterator(std::end(vals)));
    }

    spdlog::debug("[decode_samples_in_parallel] Finished decoding the output.");
}

}  // namespace dorado::polisher
