#include "polish/polish_impl.h"

#include "hts_utils/FastxRandomReader.h"
#include "secondary/architectures/model_factory.h"
#include "secondary/common/batching.h"
#include "secondary/common/region.h"
#include "secondary/consensus/sample_collate_utils.h"
#include "secondary/consensus/variant_calling.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"
#include "utils/memory_utils.h"
#include "utils/ssize.h"
#include "utils/string_utils.h"
#include "utils/timer_high_res.h"

#include <ATen/ATen.h>
#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <stdexcept>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION
// #define DEBUG_INFERENCE_DATA
// #define DEBUG_VC_DATA
// #define DEBUG_DUMP_INFERENCE_TENSORS_TO_DISK
// #define DEBUG_POLISH_SPLIT_SAMPLES_AROUND_POSITIONS
// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

#ifdef DEBUG_VC_DATA
#include "secondary/consensus/consensus_utils.h"
#endif

namespace dorado::polisher {

namespace {

std::vector<DeviceInfo> init_devices(const std::string& devices_str) {
    std::vector<DeviceInfo> devices;

    if (devices_str == "cpu") {
        torch::Device torch_device = torch::Device(devices_str);
        devices.emplace_back(DeviceInfo{.name = devices_str,
                                        .type = DeviceType::CPU,
                                        .device = std::move(torch_device),
                                        .available_memory_GB = utils::available_host_memory_GB()});
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
            const double available_memory_GB =
                    utils::available_memory(torch_device) / dorado::utils::BYTES_PER_GB;
            devices.emplace_back(DeviceInfo{.name = val,
                                            .type = DeviceType::CUDA,
                                            .device = std::move(torch_device),
                                            .available_memory_GB = available_memory_GB});
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
                                   const std::filesystem::path& in_ref_fn,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   const int32_t num_bam_threads,
                                   const int32_t num_inference_threads,
                                   const bool full_precision,
                                   const std::string& read_group,
                                   const std::string& tag_name,
                                   const int32_t tag_value,
                                   const double min_snp_accuracy,
                                   const std::optional<bool>& tag_keep_missing_override,
                                   const std::optional<int32_t>& min_mapq_override,
                                   const std::optional<secondary::HaplotagSource>& haptag_source,
                                   const std::optional<std::filesystem::path>& phasing_bin_fn,
                                   const secondary::KadayashiOptions& kadayashi_opt) {
    PolisherResources resources;

    spdlog::info("Initializing the devices.");
    resources.devices = init_devices(device_str);
    if (std::empty(resources.devices)) {
        throw std::runtime_error("Zero devices initialized! Need at least one device to run.");
    }

    spdlog::debug("Initialized devices:");
    for (int32_t device_id = 0; device_id < dorado::ssize(resources.devices); ++device_id) {
        const DeviceInfo& dev_info = resources.devices[device_id];
        spdlog::debug("    - [device_id = {}] name = {}, available_memory = {:.2f} GB", device_id,
                      dev_info.name, dev_info.available_memory_GB);
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
                    spdlog::debug("[create_resources] Converting the model to half precision.");
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

    // Open the BAM file for each thread.
    const int32_t max_num_encoders =
            std::max(num_bam_threads, static_cast<int32_t>(std::size(resources.models)));
    spdlog::info("Creating {} encoders.", max_num_encoders);
    for (int32_t i = 0; i < max_num_encoders; ++i) {
        resources.encoders.emplace_back(
                encoder_factory(model_config, in_ref_fn, in_aln_bam_fn, read_group, tag_name,
                                tag_value, true, min_snp_accuracy, tag_keep_missing_override,
                                min_mapq_override, haptag_source, phasing_bin_fn, kadayashi_opt));
    }

    spdlog::info("Creating the decoder.");
    resources.decoder = decoder_factory(model_config);

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

std::vector<std::vector<secondary::ConsensusResult>> stitch_sequence(
        const hts_io::FastxRandomReader& fastx_reader,
        const std::string& header,
        const std::vector<std::vector<secondary::ConsensusResult>>& sample_results,
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
        return {{secondary::ConsensusResult{header, draft, std::move(dummy_quals)}}};
    } else if (!fill_gaps && std::empty(samples_for_seq)) {
        spdlog::debug(
                "Sequence '{}' of length {} has zero inferred samples. NOT copying contig "
                "verbatim from input because fill_gaps == false.",
                header, std::size(draft));
        return {};
    }

    // Find the maximum number of haplotypes. All inner vectors should either be of
    // same size or empty, nothing else.
    int64_t max_haps = 0;
    for (const auto& part : sample_results) {
        max_haps = std::max(max_haps, dorado::ssize(part));
    }

    std::vector<std::vector<secondary::ConsensusResult>> ret;

    const auto init_part = [&max_haps, &header, &draft_len]() {
        std::vector<secondary::ConsensusResult> part;
        for (int64_t hap_id = 0; hap_id < max_haps; ++hap_id) {
            secondary::ConsensusResult hap_result;
            hap_result.name = header;
            hap_result.draft_start = draft_len;
            hap_result.draft_end = 0;
            part.emplace_back(hap_result);
        }
        return part;
    };

    std::vector<secondary::ConsensusResult> part = init_part();

    // This is an inclusive coordinate.
    int64_t last_end = 0;
    for (size_t i = 0; i < std::size(samples_for_seq); ++i) {
        const int32_t sample_index = samples_for_seq[i].second;
        const std::vector<secondary::ConsensusResult>& sample_haps = sample_results[sample_index];

        if (!std::empty(sample_haps) && (dorado::ssize(sample_haps) != max_haps)) {
            spdlog::warn(
                    "Unexpected number of haplotype sequences found for a sample. Expected that "
                    "all samples have the same number of generated haplotype consensus sequences, "
                    "but max_haps = {}, and number of haplotypes for the current sample: {}. "
                    "Returning empty.",
                    max_haps, std::size(sample_haps));
            return {};
        }

        // This should not happen. Create a multi-part output if so.
        if (std::empty(sample_haps)) {
            if (!std::empty(part) && !std::empty(part.front().seq)) {
                ret.emplace_back(std::move(part));
            }
            part = init_part();
            continue;
        }

        // Draft start should be identical for all haplotypes of this sample.
        const int64_t draft_start = sample_haps.front().draft_start;
        const int64_t draft_end = sample_haps.front().draft_end;

        // Fill the gap with either the draft or a fill char.
        if (draft_start > last_end) {
            if (fill_gaps) {
                const int64_t fill_len = draft_start - last_end;
                const std::string fill_seq = (fill_char) ? std::string(fill_len, *fill_char)
                                                         : draft.substr(last_end, fill_len);
                // Fill all haplotypes.
                for (secondary::ConsensusResult& hap_result : part) {
                    hap_result.seq += fill_seq;
                    hap_result.quals += std::string(fill_len, '!');
                    hap_result.draft_start = std::min(hap_result.draft_start, last_end);
                    hap_result.draft_end = std::max(hap_result.draft_end, draft_start);
                }
            } else {
                if (!std::empty(part) && !std::empty(part.front().seq)) {
                    ret.emplace_back(std::move(part));
                }
                part = init_part();
            }
        }

        // Append the sequence.
        for (int64_t hap_id = 0; hap_id < max_haps; ++hap_id) {
            const secondary::ConsensusResult& sample_result = sample_results[sample_index][hap_id];
            secondary::ConsensusResult& hap_result = part[hap_id];

            // Splice a polished chunk.
            hap_result.seq += sample_result.seq;
            hap_result.quals += sample_result.quals;
            hap_result.draft_start = std::min(hap_result.draft_start, sample_result.draft_start);
            hap_result.draft_end = std::max(hap_result.draft_end, sample_result.draft_end);
        }

        last_end = draft_end;
    }

    // Add the back draft part (or fill char).
    if ((last_end < dorado::ssize(draft)) && fill_gaps) {
        const int64_t fill_len = draft_len - last_end;
        const std::string fill_seq =
                (fill_char) ? std::string(fill_len, *fill_char) : draft.substr(last_end);
        // Fill all haplotypes.
        for (secondary::ConsensusResult& hap_result : part) {
            hap_result.seq += fill_seq;
            hap_result.quals += std::string(fill_len, '!');
            hap_result.draft_start = std::min(hap_result.draft_start, last_end);
            hap_result.draft_end = std::max(hap_result.draft_end, draft_len);
        }
        if (!std::empty(part) && !std::empty(part.front().seq)) {
            ret.emplace_back(std::move(part));
        }
    }

    spdlog::trace("[stitch_sequence] header = '{}', final.", header);

    if (!std::empty(part) && !std::empty(part.front().seq)) {
        ret.emplace_back(std::move(part));
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

    std::vector<secondary::Sample> results;
    results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        if (sample_len <= chunk_len) {
            results.emplace_back(std::move(sample));
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        // Slice out all but the last chunk unless perfectly sized.
        int64_t end = 0;
        for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
            end = start + chunk_len;
            results.emplace_back(slice_sample(sample, start, end, false));
        }

        // Last chunk will have a large overlap with previous, to maintain equal length.
        if (end < sample_len) {
            const int64_t start = sample_len - chunk_len;
            end = sample_len;
            results.emplace_back(slice_sample(sample, start, end, false));
        }
    }

    return results;
}

std::vector<secondary::Sample> split_samples_around_positions(
        std::vector<secondary::Sample> samples,
        const std::optional<IntervalTreesInt64Map>& candidate_trees,
        const int64_t chunk_len,
        const int64_t flanking_bases) {
    constexpr int64_t MIN_FLANKING_BASES = 3;

    if ((flanking_bases < 0) || (flanking_bases > chunk_len)) {
        throw std::runtime_error(
                "Wrong flanking_bases length. chunk_len = " + std::to_string(chunk_len) +
                ", flanking_bases = " + std::to_string(flanking_bases));
    }

    if (!candidate_trees) {
        return {};
    }

    const auto searchsorted_left = [](const std::vector<int64_t>& vec, const int64_t x) -> int64_t {
        const auto it = std::lower_bound(std::begin(vec), std::end(vec), x);
        return static_cast<std::int64_t>(std::distance(std::begin(vec), it));
    };

    std::vector<secondary::Sample> all_results;
    all_results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const auto it_seq_id = candidate_trees->find(sample.seq_id);

        // No candidate positions for this sequence.
        if (it_seq_id == std::cend(*candidate_trees)) {
            continue;
        }
        const auto& tree = it_seq_id->second;

        // Get all candidate positions for this region.
        // Note: this interval tree lib uses inclusive end coordinate.
        std::vector<interval_tree::Interval<int64_t, int64_t>> positions =
                tree.findOverlapping(sample.start(), sample.end() - 1);

        // Sort the positions in ascending order.
        std::sort(std::begin(positions), std::end(positions),
                  [](const auto& a, const auto& b) { return a.start < b.start; });

        std::vector<secondary::Sample> results;
        std::vector<int64_t> last_flanking_bases;

#ifdef DEBUG_POLISH_SPLIT_SAMPLES_AROUND_POSITIONS
        spdlog::debug("[split_samples_around_positions] Input sample: {}",
                      secondary::sample_to_string(sample));
        for (int64_t i = 0; i < std::ssize(positions); ++i) {
            spdlog::debug("[split_samples_around_positions]     [candidate i = {}] position = {}",
                          i, positions[i].start);
        }
#endif

        bool stop_chunking = false;
        for (const auto itvl : positions) {
            // Intervals are single-base width here.
            const int64_t position = itvl.start;

            // Skip candidates which are already covered by the previous chunk.
            assert(std::size(last_flanking_bases) == std::size(results));
            if (!std::empty(results) && !std::empty(last_flanking_bases) &&
                (position < (results.back().end() - last_flanking_bases.back()))) {
                continue;
            }

            int64_t curr_flanking_bases = 0;
            int64_t chunk_start_pos = 0;
            int64_t chunk_start_idx = 0;
            int64_t chunk_end_idx = 0;
            for (curr_flanking_bases = flanking_bases; curr_flanking_bases >= MIN_FLANKING_BASES;
                 --curr_flanking_bases) {
                chunk_start_pos = position - curr_flanking_bases;
                chunk_start_idx = searchsorted_left(sample.positions_major, chunk_start_pos);
                chunk_end_idx = chunk_start_idx + chunk_len;

                if (chunk_start_idx >= std::ssize(sample.positions_major)) {
                    // This shouldn't be possible, but need to check the bounds.
                    std::ostringstream oss;
                    oss << "Tried to create chunk from chunk_start_pos = " << chunk_start_pos
                        << " on seq_id = " << sample.seq_id
                        << " but the position could not be found in this sample! chunk_start_idx = "
                        << chunk_start_idx << ", sample: " << sample;
                    throw std::runtime_error{oss.str()};
                }

                // TODO: Handle this properly. E.g. Create an overlapping large chunk at the end.
                if (chunk_end_idx >= std::ssize(sample.positions_major)) {
                    spdlog::warn(
                            "Tried to create chunk but "
                            "chunk exceeds sample length. chunk_start_pos = {}, chunk_start_idx = "
                            "{}, chunk_end_idx = {}, positions_major.size() = {}, seq_id = {}. "
                            "Stopping. Sample: {}",
                            chunk_start_pos, chunk_start_idx, chunk_end_idx,
                            std::ssize(sample.positions_major), sample.seq_id, chunk_end_idx,
                            secondary::sample_to_string(sample));
                    stop_chunking = true;
                    break;
                }

                const int64_t chunk_end_pos = sample.positions_major[chunk_end_idx];

                if (position <= (chunk_end_pos - curr_flanking_bases)) {
                    break;
                }
            }
            if (stop_chunking) {
                break;
            }
            if (curr_flanking_bases < MIN_FLANKING_BASES) {
                spdlog::warn(
                        "Could not create chunk around position = {} with more than {} bases of "
                        "flanking context. Skipping this position.",
                        position, MIN_FLANKING_BASES);
                continue;
            }

            // Extract the chunk around this position.
            results.emplace_back(slice_sample(sample, chunk_start_idx, chunk_end_idx, true));
            last_flanking_bases.emplace_back(curr_flanking_bases);

#ifdef DEBUG_POLISH_SPLIT_SAMPLES_AROUND_POSITIONS
            spdlog::debug(
                    "[split_samples_around_positions] Created a chunk around position = {}. "
                    "chunk_start_pos = {}, chunk_start_idx = {}, chunk_end_idx = {}, "
                    "curr_flanking_bases = {}. Chunk sample: {}",
                    position, chunk_start_pos, chunk_start_idx, chunk_end_idx, curr_flanking_bases,
                    secondary::sample_to_string(results.back()));
#endif
        }

        all_results.insert(std::end(all_results), std::make_move_iterator(std::begin(results)),
                           std::make_move_iterator(std::end(results)));
    }

    return all_results;
}

std::vector<secondary::Sample> split_samples_tiled_with_candidates(
        std::vector<secondary::Sample> samples,
        const std::optional<IntervalTreesInt64Map>& candidate_trees,
        const int64_t chunk_len,
        const int64_t chunk_overlap,
        const bool ext_flanks,          // Control extension heuristic.
        const int64_t ext_major_bases,  // Check this many major positions to trigger.
        const int64_t ext_min_cov,      // Minimum absolute coverage to trigger the heuristic.
        const double ext_cov_frac       // Minimum coverage fraction to trigger the heuristic.
) {
    if ((chunk_overlap < 0) || (chunk_overlap > chunk_len)) {
        throw std::runtime_error(
                "Wrong chunk_overlap length. chunk_len = " + std::to_string(chunk_len) +
                ", chunk_overlap = " + std::to_string(chunk_overlap));
    }

    const auto has_candidates = [](const dorado::polisher::IntervalTreeInt64& tree,
                                   const secondary::Sample& sample, const int64_t start_idx,
                                   const int64_t end_idx) {
        if ((start_idx < 0) || (end_idx <= 0) || (start_idx >= end_idx) ||
            (end_idx > std::ssize(sample.positions_major))) {
            return false;
        }
        const int64_t start = sample.positions_major[start_idx];
        const int64_t end = sample.positions_major[end_idx - 1];  // Inclusive for IntervalTree.
        const std::vector<interval_tree::Interval<int64_t, int64_t>> positions =
                tree.findOverlapping(start, end);
        return !std::empty(positions);
    };

    const auto check_excess_deletions = [](const secondary::Sample& sample, const int64_t start_idx,
                                           const int64_t end_idx, const bool reverse,
                                           const int64_t major_bases, const double cov_fraction,
                                           const int64_t min_abs_cov) {
        /// @brief Returns true if any of the first/last `major_bases` major positions have many deletion
        ///         counts (above `max(cov * cov_fraction, min_abs_cov)`).
        static constexpr int8_t DEL_VAL = 5;  // Value representing deletion in base channel.

        const int64_t cov = sample.features.size(1);
        const int64_t min_count = std::max(min_abs_cov, static_cast<int64_t>(cov * cov_fraction));

        if (!reverse) {
            for (int64_t pos = start_idx, num_major = 0; pos < end_idx; ++pos) {
                if (sample.positions_minor[pos] > 0) {
                    continue;
                }
                ++num_major;
                if (num_major > major_bases) {
                    break;
                }
                const at::Tensor pos_slice = sample.features.index({pos});
                const at::Tensor bases = pos_slice.index({torch::indexing::Slice(), 0});
                const at::Tensor mask = (bases == DEL_VAL);
                const int64_t count = mask.sum().item<int64_t>();
                if (count >= min_count) {
                    return true;
                }
            }
        } else {
            for (int64_t pos = (end_idx - 1), num_major = 0; pos >= start_idx; --pos) {
                if (sample.positions_minor[pos] > 0) {
                    continue;
                }
                ++num_major;
                if (num_major > major_bases) {
                    break;
                }
                const at::Tensor pos_slice = sample.features.index({pos});
                const at::Tensor bases = pos_slice.index({torch::indexing::Slice(), 0});
                const at::Tensor mask = (bases == DEL_VAL);
                const int64_t count = mask.sum().item<int64_t>();
                if (count >= min_count) {
                    return true;
                }
            }
        }
        return false;
    };

    const auto check_flanking_minor = [](const secondary::Sample& sample, const int64_t start_idx,
                                         const int64_t end_idx, const bool reverse) {
        /// @brief Returns true if the first/last position is a minor one.
        if (std::empty(sample.positions_minor)) {
            return false;
        }
        if (start_idx >= end_idx) {
            return false;
        }
        if ((start_idx < 0) || (end_idx <= 0) ||
            (start_idx >= std::ssize(sample.positions_minor)) ||
            (end_idx > std::ssize(sample.positions_minor))) {
            return false;
        }
        if (!reverse) {
            return sample.positions_minor[start_idx] != 0;
        } else {
            return sample.positions_minor[end_idx - 1] != 0;
        }
        return false;
    };

    std::vector<secondary::Sample> results;
    results.reserve(std::size(samples));

    for (auto& sample : samples) {
        const int64_t sample_len = static_cast<int64_t>(std::size(sample.positions_major));

        // Get the interval tree of candidates for this sequence ID.
        const auto it_seq_id = candidate_trees->find(sample.seq_id);
        if (it_seq_id == std::cend(*candidate_trees)) {
            spdlog::debug("Cannot find seq_id = {} in candidate_trees! Sample: {}", sample.seq_id,
                          secondary::sample_to_string(sample));
            continue;
        }
        const auto& tree = it_seq_id->second;

        if (sample_len <= chunk_len) {
            if (has_candidates(tree, sample, 0, sample_len)) {
                results.emplace_back(std::move(sample));
            }
            continue;
        }

        const int64_t step = chunk_len - chunk_overlap;

        // Create window coordinates.
        std::vector<std::pair<int64_t, int64_t>> windows;
        {
            windows.reserve((sample_len - chunk_len) / chunk_overlap);
            int64_t end = 0;
            for (int64_t start = 0; start < (sample_len - chunk_len + 1); start += step) {
                end = start + chunk_len;
                windows.emplace_back(start, end);
            }
            if (end < sample_len) {
                const int64_t start = sample_len - chunk_len;
                end = sample_len;
                windows.emplace_back(start, end);
            }
        }

        // Find windows with candidates.
        std::vector<bool> window_used(std::size(windows), false);
        for (int64_t i = 0; i < std::ssize(windows); ++i) {
            const auto [start, end] = windows[i];
            if (has_candidates(tree, sample, start, end)) {
                window_used[i] = true;
            }
        }

        // Heuristic to include neighboring windows if there are possible
        // deletions at the flanks.
        if (ext_flanks) {
            for (int64_t i = 0; i < std::ssize(windows); ++i) {
                if (!window_used[i]) {
                    continue;
                }

                // Extend to the left if needed.
                for (int64_t j = i; j > 0; --j) {
                    // Predecessor window is already used, no need to extend to the left any further.
                    if (!window_used[j] || window_used[j - 1]) {
                        break;
                    }
                    const auto [start, end] = windows[j];
                    if (!check_excess_deletions(sample, start, end, false, ext_major_bases,
                                                ext_cov_frac, ext_min_cov) ||
                        !check_flanking_minor(sample, start, end, false)) {
                        break;
                    }
                    window_used[j - 1] = true;
                }

                // Extend to the right if needed.
                for (int64_t j = i; j < (std::ssize(windows) - 1); ++j) {
                    // Next window is already used, no need to extend to the right any further.
                    if (!window_used[j] || window_used[j + 1]) {
                        break;
                    }
                    const auto [start, end] = windows[j];
                    if (!check_excess_deletions(sample, start, end, true, ext_major_bases,
                                                ext_cov_frac, ext_min_cov) ||
                        !check_flanking_minor(sample, start, end, true)) {
                        break;
                    }
                    window_used[j + 1] = true;
                }
            }
        }

        // Slice out selected windows to return.
        for (int64_t i = 0; i < std::ssize(windows); ++i) {
            if (window_used[i]) {
                const auto [start, end] = windows[i];
                results.emplace_back(slice_sample(sample, start, end, true));
            }
        }
    }

    return results;
}
/**
 * \brief This function performs the following operations:
 *          1. Merges adjacent samples, which were split for efficiency of computing the pileup.
 *          2. Checks for discontinuities in any of the samples (based on major positions) and splits them.
 *          3. Splits the merged samples into equally sized pieces which will be used for inference to prevent memory usage spikes.
 * \param window_samples Input samples which will be merged and split. Non-const to enable moving of data.
 * \param encoder Encoder used to produce the sample tensors. It is needed becaue of the secondary::EncoderBase::merge_adjacent_samples() function.
 * \param bam_regions BAM region coordinates. This is a Span to facilitate batching of BAM regions from the outside.
 * \param bam_region_intervals Range of IDs of window_samples which comprise this BAM region. E.g. BAM region 0 uses window_samples[0:5], BAM region 1 uses window_samples[5:9], etc.
 *                              This is a Span to facilitate batching of BAM regions from the outside and avoid copying vectors.
 * \param num_threads Number of threads for procesing.
 * \param window_len Length of the window to split the final samples into.
 * \param window_overlap Overlap between neighboring windows when splitting.
 * \param window_interval_offset Used for batching bam_region_intervals, because window_samples.size() matches the total size of bam_region_intervals,
 *                                  while coordinates of each BAM region interval are global and produced before draft batching on the client side.
 */
std::pair<std::vector<secondary::Sample>, std::vector<secondary::TrimInfo>>
merge_and_split_bam_regions_in_parallel(
        std::vector<secondary::Sample>& window_samples,
        std::atomic<bool>& worker_terminate,
        const std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
        const std::span<const secondary::Window> bam_regions,
        const std::span<const secondary::Interval> bam_region_intervals,
        const std::optional<IntervalTreesInt64Map>& candidate_trees,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t variant_flanking_bases,
        const int32_t window_interval_offset,
        const bool continue_on_exception,
        const bool tiled_regions,
        const bool tiled_ext_flanks,
        const int64_t tiled_ext_major,
        const int64_t tiled_ext_min_cov,
        const float tiled_ext_cov_fract) {
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

    const auto worker = [&](const int32_t tid, const int32_t start, const int32_t end,
                            std::vector<std::vector<secondary::Sample>>& results_samples,
                            std::vector<std::vector<secondary::TrimInfo>>& results_trims,
                            WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("merge_and_split_bam_regions_in_parallel-worker", 4);

        for (int32_t bam_region_id = start; bam_region_id < end; ++bam_region_id) {
            if (worker_terminate) {
                return;
            }

            try {
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
                local_samples = encoders[tid]->merge_adjacent_samples(local_samples);

                spdlog::trace(
                        "- [bam_region_id = {}] (2) After merging adjacent: local_samples = {}",
                        bam_region_id, std::size(local_samples));
#ifdef DEBUG_POLISH_SAMPLE_CONSTRUCTION
                debug_print_samples(std::cerr, local_samples, 0, -1, -1);
#endif

                // Bluntly split samples for inference.
                if (candidate_trees) {
                    if (!tiled_regions) {
                        local_samples = split_samples_around_positions(std::move(local_samples),
                                                                       candidate_trees, window_len,
                                                                       variant_flanking_bases);
                    } else {
                        local_samples = split_samples_tiled_with_candidates(
                                std::move(local_samples), candidate_trees, window_len,
                                window_overlap, tiled_ext_flanks, tiled_ext_major,
                                tiled_ext_min_cov, tiled_ext_cov_fract);
                    }
                } else {
                    local_samples =
                            split_samples(std::move(local_samples), window_len, window_overlap);
                }

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
                        local_samples,
                        std::optional<secondary::RegionInt>(
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
            } catch (const std::exception& e) {
                if (continue_on_exception) {
                    spdlog::warn("Caught exception in merge_and_split_bam_regions: '{}'", e.what());
                } else {
                    ret_val = {.exception_thrown = true,
                               .message = "Caught exception in merge_and_split_bam_regions: '" +
                                          std::string(e.what()) + "'"};
                    worker_terminate = true;
                    return;
                }
            }
        }
    };

    // Result vectors for each BAM region.
    std::vector<std::vector<secondary::Sample>> merged_samples(std::size(bam_region_intervals));
    std::vector<std::vector<secondary::TrimInfo>> merged_trims(std::size(bam_region_intervals));

    // Process BAM regions in parallel.
    const int32_t max_num_threads =
            std::min(num_threads, static_cast<int32_t>(std::size(encoders)));
    const std::vector<secondary::Interval> thread_chunks = secondary::compute_partitions(
            static_cast<int32_t>(std::size(bam_region_intervals)), max_num_threads);

    spdlog::trace("Starting to merge samples for {} BAM windows using {} threads.",
                  std::size(bam_region_intervals), std::size(thread_chunks));

    // Parallel processing of BAM regions.
    cxxpool::thread_pool pool{std::size(thread_chunks)};
    std::vector<std::future<void>> futures;
    futures.reserve(std::size(thread_chunks));
    std::vector<WorkerReturnStatus> worker_return_vals(std::size(thread_chunks));
    for (size_t tid = 0; tid < std::size(thread_chunks); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, tid, chunk_start, chunk_end,
                                       std::ref(merged_samples), std::ref(merged_trims),
                                       std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    for (size_t tid = 0; tid < std::size(worker_return_vals); ++tid) {
        const WorkerReturnStatus& rv = worker_return_vals[tid];
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            throw std::runtime_error{"(merge-samples) " + rv.message};
        } else {
            spdlog::warn("(merge-samples) " + rv.message);
        }
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

/**
 * \brief For each input window (region of the draft) runs the given encoder and produces a sample.
 *          The BamFile handels are used to fetch the pileup data and encode regions.
 *          Encoding is parallelized, where the actual number of threads is min(bam_handles.size(), num_threads, windows.size()).
 */
std::vector<secondary::Sample> encode_windows_in_parallel(
        std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
        std::atomic<bool>& worker_terminate,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<std::unordered_map<std::string, int32_t>>& bam_region_haplotags,
        const std::span<const secondary::Window> windows,
        const int32_t num_threads,
        const bool continue_on_exception) {
    utils::ScopedProfileRange spr1("encode_windows_in_parallel", 3);

    const std::unordered_map<std::string, int32_t> empty_haplotags;

    // Find the haplotags for this BAM region.
    const auto find_haplotags_for_region = [&bam_region_haplotags,
                                            &empty_haplotags](const int32_t bam_region_id)
            -> const std::unordered_map<std::string, int32_t>& {
        // If the input is unphased, just return an empty set.
        if (std::empty(bam_region_haplotags)) {
            return empty_haplotags;
        }

        // Otherwise, get the haplotags for this region and validate the indices.
        const bool bam_region_valid =
                (bam_region_id >= 0) && (bam_region_id < std::ssize(bam_region_haplotags));

        if (!bam_region_valid) {
            throw std::logic_error{"The window.source_region_id is not valid! Value = " +
                                   std::to_string(bam_region_id) +
                                   ", bam_region_haplotags.size() = " +
                                   std::to_string(std::size(bam_region_haplotags))};
        }

        return bam_region_haplotags[bam_region_id];
    };

    // Worker function, each thread computes tensors for a set of windows assigned to it.
    const auto worker = [&](const int32_t thread_id, utils::AsyncQueue<std::size_t>& window_queue,
                            std::vector<secondary::Sample>& results, WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("encode_windows_in_parallel-worker", 4);

        const std::size_t n_windows = std::size(windows);
        std::size_t window_id = 0;

        while (window_queue.try_pop(window_id) != utils::AsyncQueueStatus::Terminate) {
            if (worker_terminate) {
                window_queue.terminate(dorado::utils::AsyncQueueTerminateFast::Yes);
                return;
            }

            try {
                const auto& window = windows[window_id];

                // Find the haplotags for this BAM region.
                const std::unordered_map<std::string, int32_t>& haplotags =
                        find_haplotags_for_region(window.source_region_id);

                const std::string& name = draft_lens[window.seq_id].first;

                if (thread_id == 0) {
                    spdlog::trace(
                            "[encoder {}] encoding window_id = {}, region = "
                            "{}:{}-{} ({} %).",
                            thread_id, window_id, name, window.start, window.end,
                            100.0 * static_cast<double>(window_id) / n_windows);
                }

                results[window_id] = encoders[thread_id]->encode_region(
                        name, window.start, window.end, window.seq_id, haplotags);

            } catch (const std::exception& e) {
                if (continue_on_exception) {
                    spdlog::warn(e.what());
                } else {
                    ret_val = {.exception_thrown = true, .message = e.what()};
                    worker_terminate = true;
                    return;
                }
            }
        }
    };

    // Initialize a shared async queue with window IDs which threads can pop.
    utils::AsyncQueue<std::size_t> shared_window_queue(std::size(windows));
    for (std::size_t i = 0; i < std::size(windows); ++i) {
        shared_window_queue.try_push(std::move(i));
    }
    shared_window_queue.terminate(utils::AsyncQueueTerminateFast::No);

    // Create the thread pool, futures and results.
    const std::size_t actual_threads =
            std::min(num_threads, static_cast<int32_t>(std::size(encoders)));
    cxxpool::thread_pool pool{actual_threads};
    std::vector<std::future<void>> futures;
    futures.reserve(actual_threads);
    std::vector<secondary::Sample> results(std::size(windows));
    std::vector<WorkerReturnStatus> worker_return_vals(actual_threads);

    spdlog::debug("Starting to encode regions for {} windows using {} threads.", std::size(windows),
                  actual_threads);

    // Add jobs to the pool.
    for (int32_t tid = 0; tid < static_cast<int32_t>(actual_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(shared_window_queue),
                                       std::ref(results), std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    for (size_t tid = 0; tid < std::size(worker_return_vals); ++tid) {
        const WorkerReturnStatus& rv = worker_return_vals[tid];
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            throw std::runtime_error{rv.message};
        } else {
            spdlog::warn(rv.message);
        }
    }

    return results;
}

}  // namespace

std::vector<secondary::Window> create_windows_from_regions(
        const std::vector<secondary::Region>& regions,
        const std::unordered_map<std::string, std::pair<int64_t, int64_t>>& draft_lookup,
        const int32_t bam_chunk_len,
        const int32_t window_overlap) {
    utils::ScopedProfileRange spr1("create_windows_from_regions", 2);

    std::vector<secondary::Window> windows;

    for (int64_t i = 0; i < std::ssize(regions); ++i) {
        secondary::Region region = regions[i];

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
        std::vector<secondary::Window> new_windows = secondary::create_windows(
                static_cast<int32_t>(seq_id), region.start, region.end, seq_length, bam_chunk_len,
                window_overlap, static_cast<int32_t>(i));

        spdlog::debug("Generated {} windows for region: '{}'.", std::size(new_windows),
                      region_to_string(region));
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    return windows;
}

std::vector<kadayashi::varcall_result_t> haplotag_regions_in_parallel(
        const std::vector<secondary::Window>& regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
        const int32_t num_threads) {
    if (std::empty(encoders)) {
        return {};
    }

    // Result data.
    std::vector<kadayashi::varcall_result_t> region_haplotags(std::size(regions));

    // Counters to process the queue.
    const int64_t num_regions = std::ssize(regions);
    std::atomic<int64_t> num_processed{0};

    const auto worker = [&](const int32_t thread_id) {
        auto& encoder = *encoders[thread_id];

        while (true) {
            // Fetch next job.
            const int64_t region_id = num_processed.fetch_add(1, std::memory_order_relaxed);
            if (region_id >= num_regions) {
                break;
            }

            // Get the region info.
            const secondary::Window& region = regions[region_id];
            const std::string& ref_name = draft_lens[region.seq_id].first;

            // Haplotag.
            region_haplotags[region_id] =
                    (encoder.produce_haplotags(ref_name, region.start, region.end));
        }
    };

    // Thread pool.
    const size_t final_num_threads =
            std::min(static_cast<size_t>(num_threads), std::size(encoders));
    cxxpool::thread_pool pool{final_num_threads};
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);
    for (int32_t tid = 0; tid < static_cast<int32_t>(final_num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid));
    }
    for (auto& f : futures) {
        f.get();
    }

    return region_haplotags;
}

void sample_producer(
        PolisherResources& resources,
        const std::vector<secondary::Window>& bam_regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<std::unordered_map<std::string, int32_t>>& bam_region_haplotags,
        const std::optional<IntervalTreesInt64Map>& candidate_trees,
        const int32_t num_threads,
        const int32_t batch_size,
        const int32_t encoding_batch_size,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t variant_flanking_bases,
        const int32_t bam_subchunk_len,
        const double max_available_mem,
        const bool continue_on_exception,
        const bool tiled_regions,
        const bool tiled_ext_flanks,
        const int64_t tiled_ext_major,
        const int64_t tiled_ext_min_cov,
        const float tiled_ext_cov_fract,
        utils::AsyncQueue<InferenceData>& infer_data,
        std::atomic<bool>& worker_terminate,
        WorkerReturnStatus& ret_status) {
    utils::ScopedProfileRange spr1("sample_producer", 2);

    spdlog::debug("[producer] Input: {} BAM windows.", std::size(bam_regions));

    const auto move_buffer_data_to_queue = [](InferenceData& buffer,
                                              utils::AsyncQueue<InferenceData>& queue,
                                              const bool any_batch_size) {
        if (std::empty(buffer.samples)) {
            return;
        }

        // Any batch size is fine (no need to find a multiple of 8).
        if (any_batch_size) {
            queue.try_push(std::move(buffer));
            buffer = {};
            return;
        }

        // Round to the nearest smaller multiple of 8. If number of samples < 8 just use what
        // there is, there is no real benefit of rounding to anything below it really.
        const int64_t num_samples = dorado::ssize(buffer.samples);
        const int64_t new_batch_size = (num_samples < 8) ? num_samples : (8 * (num_samples / 8));

        // Get the first batch_size elements and push them to the queue.
        InferenceData new_buffer;
        for (int64_t i = 0; i < new_batch_size; ++i) {
            new_buffer.samples.emplace_back(std::move(buffer.samples[i]));
            new_buffer.trims.emplace_back(std::move(buffer.trims[i]));
        }
        queue.try_push(std::move(new_buffer));

        // Get the remaining items and update the buffer.
        InferenceData remainder;
        for (int64_t i = new_batch_size; i < dorado::ssize(buffer.samples); ++i) {
            remainder.samples.emplace_back(std::move(buffer.samples[i]));
            remainder.trims.emplace_back(std::move(buffer.trims[i]));
        }
        buffer.samples = std::move(remainder.samples);
        buffer.trims = std::move(remainder.trims);
    };

    // Split large BAM regions into non-overlapping windows for parallel encoding.
    // The non-overlapping windows will be merged after samples are constructed.
    std::vector<secondary::Window> windows;
    std::vector<secondary::Interval>
            bam_region_intervals;  // Intervals of windows for each BAM region.
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(bam_regions)); ++i) {
        const secondary::Window& bw = bam_regions[i];
        std::vector<secondary::Window> new_windows = secondary::create_windows(
                bw.seq_id, bw.start, bw.end, bw.seq_length, bam_subchunk_len, 0, i);
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

    // Divide BAM regions into groups of specified size (in terms of num windows), as sort of a barrier.
    const std::vector<secondary::Interval> bam_region_batches = secondary::create_batches(
            bam_region_intervals, encoding_batch_size,
            [](const secondary::Interval& val) { return val.end - val.start; });

    InferenceData buffer;

    // All models should have the same architecture (they are just copies on different devices),
    // so get the first one to be able to reach the `estimate_batch_memory()` function.
    assert(!std::empty(resources.models));
    const auto& model = resources.models.front();

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
                    resources.encoders, worker_terminate, draft_lens, bam_region_haplotags,
                    std::span<const secondary::Window>(std::data(windows) + window_id_start,
                                                       num_windows),
                    num_threads, continue_on_exception);

            spdlog::trace(
                    "[producer] Merging the samples into {} BAM chunks. parallel_results.size() = "
                    "{}",
                    num_regions, std::size(region_samples));

            // Passing only one const encoder because it only calls const functions without sideeffects.
            auto [samples, trims] = merge_and_split_bam_regions_in_parallel(
                    region_samples, worker_terminate, resources.encoders,
                    std::span<const secondary::Window>(std::data(bam_regions) + region_id_start,
                                                       num_regions),
                    std::span<const secondary::Interval>(
                            std::data(bam_region_intervals) + region_id_start, num_regions),
                    candidate_trees, num_threads, window_len, window_overlap,
                    variant_flanking_bases, window_id_start, continue_on_exception, tiled_regions,
                    tiled_ext_flanks, tiled_ext_major, tiled_ext_min_cov, tiled_ext_cov_fract);

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
                    remainder_buffer.samples.emplace_back(std::move(samples[i]));
                    remainder_buffer.trims.emplace_back(std::move(trims[i]));
                    spdlog::trace(
                            "[producer] Pushing a remainder batch of data to infer_data queue. "
                            "remainder_buffer.samples.size() = {}. i = {}, size(positions_major) = "
                            "{}, window_len = {}, size(samples) = {}",
                            std::size(remainder_buffer.samples), i,
                            std::size(samples[i].positions_major), window_len, std::size(samples));
                    infer_data.try_push(std::move(remainder_buffer));
                    continue;
                }

                const std::vector<int64_t> curr_batch_shape =
                        secondary::compute_collated_padded_shape(buffer.samples);
                const double estimated_memory_curr =
                        std::empty(curr_batch_shape)
                                ? 0.0
                                : model->estimate_batch_memory(curr_batch_shape);

                // Cut batches either on memory consumption or on the absolute count.
                if (batch_size <= 0) {
                    // Auto batch size computation.
                    const std::vector<int64_t> next_batch_shape =
                            secondary::compute_collated_padded_shape(buffer.samples, samples[i]);

                    const double estimated_memory_next =
                            model->estimate_batch_memory(next_batch_shape);

                    if (!std::empty(buffer.samples) &&
                        (estimated_memory_next >= max_available_mem)) {
                        spdlog::trace("[producer] Estimating batch memory for auto batch size:");
                        spdlog::trace("    - max_available_mem = {} GB", max_available_mem);
                        spdlog::trace("    - estimated_memory_next = {} GB", estimated_memory_next);
                        spdlog::trace(
                                "    - next_batch_shape = {}",
                                utils::print_container_as_string(next_batch_shape, ",", true));
                        spdlog::trace("    - estimated_memory_curr = {} GB", estimated_memory_curr);
                        spdlog::trace(
                                "    - curr_batch_shape = {}",
                                utils::print_container_as_string(curr_batch_shape, ",", true));

                        spdlog::trace(
                                "[producer] Pushing a batch of data to infer_data queue (1a). "
                                "buffer.samples.size() = {}",
                                std::size(buffer.samples));

                        move_buffer_data_to_queue(buffer, infer_data, false);
                    }

                } else {
                    if (dorado::ssize(buffer.samples) >= batch_size) {
                        // Fixed batch size.
                        spdlog::trace("[producer] Estimating batch memory for fixed batch size:");
                        spdlog::trace("    - max_available_mem = {} GB", max_available_mem);
                        spdlog::trace("    - estimated_memory_curr = {} GB", estimated_memory_curr);
                        spdlog::trace(
                                "    - curr_batch_shape = {}",
                                utils::print_container_as_string(curr_batch_shape, ",", true));

                        spdlog::trace(
                                "[producer] Pushing a batch of data to infer_data queue (1b). "
                                "buffer.samples.size() = {}",
                                std::size(buffer.samples));

                        move_buffer_data_to_queue(buffer, infer_data, true);
                    }
                }

                // Expand the current buffer.
                buffer.samples.emplace_back(std::move(samples[i]));
                buffer.trims.emplace_back(std::move(trims[i]));
            }
        } catch (const std::exception& e) {
            if (!continue_on_exception) {
                // Cannot throw because this is a worker function intended to run on a separate thread.
                // Instead, communicate the error and return.
                ret_status = {.exception_thrown = true, .message = e.what()};
                worker_terminate = true;
                infer_data.terminate(utils::AsyncQueueTerminateFast::No);
                return;
            }

            spdlog::warn(
                    "Caught exception in the sample producer. Skipping the remaining regions "
                    "in "
                    "current batch (region_id_start = {}, region_id_end = {}). Exception: {}",
                    region_id_start, region_id_end, e.what());
            continue;
        }
    }

    if (!std::empty(buffer.samples)) {
        const std::vector<int64_t> curr_batch_shape =
                secondary::compute_collated_padded_shape(buffer.samples);
        const double estimated_memory_curr =
                std::empty(curr_batch_shape) ? 0.0 : model->estimate_batch_memory(curr_batch_shape);

        spdlog::trace("[producer] Estimating batch memory for the final batch:");
        spdlog::trace("    - max_available_mem = {} GB", max_available_mem);
        spdlog::trace("    - estimated_memory_curr = {} GB", estimated_memory_curr);
        spdlog::trace("    - curr_batch_shape = {}",
                      utils::print_container_as_string(curr_batch_shape, ",", true));

        spdlog::trace(
                "[producer] Pushing a batch of data to infer_data queue (2). "
                "buffer.samples.size() = {} (final)",
                std::size(buffer.samples));

        // First, push the N*8 elements to the queue for efficiency.
        move_buffer_data_to_queue(buffer, infer_data, false);

        // Next, push the remainder if the buffer is not empty.
        move_buffer_data_to_queue(buffer, infer_data, true);

        spdlog::debug("[producer] Pushed final batch for inference to infer_data queue.");
    }

    infer_data.terminate(utils::AsyncQueueTerminateFast::No);
}

void infer_samples_in_parallel(
        utils::AsyncQueue<InferenceData>& batch_queue,
        utils::AsyncQueue<DecodeData>& decode_queue,
        std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
        std::atomic<bool>& worker_terminate,
        const std::vector<c10::optional<c10::Stream>>& streams,
        const std::vector<std::unique_ptr<secondary::EncoderBase>>& encoders,
        [[maybe_unused]] const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool continue_on_exception) {
    utils::ScopedProfileRange spr1("infer_samples_in_parallel", 2);

    if (std::empty(models)) {
        throw std::runtime_error("No models have been initialized, cannot run inference.");
    }

    auto batch_infer = [&encoders, &draft_lens](secondary::ModelTorchBase& model,
                                                const InferenceData& batch, const int32_t tid) {
        utils::ScopedProfileRange spr2("infer_samples_in_parallel-batch_infer", 3);
        timer::TimerHighRes timer_total;

        (void)draft_lens;

#ifdef DEBUG_DUMP_INFERENCE_TENSORS_TO_DISK
        // Debug write tensors for each sample, individually.
        {
            for (int64_t ii = 0; ii < dorado::ssize(batch.samples); ++ii) {
                const int64_t seq_id = batch.samples[ii].seq_id;
                const std::string& seq_name = draft_lens[seq_id].first;
                const int64_t s = batch.samples[ii].start();
                const int64_t e = batch.samples[ii].end();
                utils::save_tensor(batch.samples[ii].features,
                                   "debug.tensor_in.seq_" + seq_name + "." + std::to_string(s) +
                                           "_" + std::to_string(e) + ".pt");
            }
        }
#endif

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
            batch_features_tensor = encoders[tid]->collate(std::move(batch_features));
            time_collate = timer_collate.GetElapsedMilliseconds();
        }

        const std::string input_batch_tensor_shape =
                utils::tensor_shape_as_string(batch_features_tensor);

        // Debug output.
        {
            spdlog::trace(
                    "[consumer {}] About to call forward(): batch_features_tensor.size() = [{}], "
                    "approx "
                    "size: {} MB.",
                    tid, input_batch_tensor_shape,
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

#ifdef DEBUG_INFERENCE_DATA
            {
                std::cout << "[infer] input: batch_features_tensor.shape = "
                          << utils::tensor_shape_as_string(batch_features_tensor) << "\n";
                std::cout << "[infer] input: batch_features_tensor =\n"
                          << batch_features_tensor << "\n";
                utils::save_tensor(batch_features_tensor, "debug.tensor.in.pt");
            }
#endif

            try {
                output = model.predict_on_batch(std::move(batch_features_tensor));
            } catch (const std::exception& e) {
                spdlog::error("Exception caught: {}", e.what());
                throw;
            }

#ifdef DEBUG_INFERENCE_DATA
            {
                std::cout << "[infer] output: output.shape = "
                          << utils::tensor_shape_as_string(output) << "\n";
                std::cout << "[infer] output: output =\n" << output << "\n";
                utils::save_tensor(output, "debug.tensor.out.pt");
            }
#endif
        }

#ifdef DEBUG_DUMP_INFERENCE_TENSORS_TO_DISK
        // Debug write output tensors for each sample, individually.
        {
            for (int64_t ii = 0; ii < output.size(0); ++ii) {
                const int64_t seq_id = batch.samples[ii].seq_id;
                const std::string& seq_name = draft_lens[seq_id].first;
                const int64_t s = batch.samples[ii].start();
                const int64_t e = batch.samples[ii].end();
                utils::save_tensor(output[ii], "debug.tensor_out.seq_" + seq_name + "." +
                                                       std::to_string(s) + "_" + std::to_string(e) +
                                                       ".pt");
            }
        }
#endif

        // Debug output.
        {
            const int64_t time_forward = timer_forward.GetElapsedMilliseconds();
            const int64_t time_total = timer_total.GetElapsedMilliseconds();

            spdlog::trace(
                    "[consumer {}] Computed batch inference. Timings - collate: {} ms, forward: {} "
                    "ms, "
                    "total = {}, batch_features_tensor.shape = [{}]",
                    tid, time_collate, time_forward, time_total, input_batch_tensor_shape);
        }

        return output;
    };

    const auto worker = [&](const int32_t tid, secondary::ModelTorchBase& model,
                            [[maybe_unused]] const c10::optional<c10::Stream>& stream,
                            WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("infer_samples_in_parallel-worker", 3);

#if DORADO_CUDA_BUILD
        c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

        at::InferenceMode infer_guard;

        while (!worker_terminate) {
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
                if (continue_on_exception) {
                    spdlog::warn(
                            "Caught exception while inferring a batch of samples: '{}'. Skipping "
                            "this batch.",
                            e.what());
                } else {
                    ret_val = {.exception_thrown = true,
                               .message = "Caught exception while inferring a batch of samples: '" +
                                          std::string(e.what()) + "'"};
                    worker_terminate = true;
                    return;
                }
            }
        }
    };

    if (std::size(models) > std::size(encoders)) {
        spdlog::warn("There are more models than there are encoders! Num models: " +
                     std::to_string(std::size(models)) + ", num encoders: " +
                     std::to_string(std::size(encoders)) + ". Using fewer models.");
    }

    const size_t num_threads = std::min(std::size(models), std::size(encoders));
    cxxpool::thread_pool pool{num_threads};

    std::vector<WorkerReturnStatus> worker_return_vals(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(*models[tid]), streams[tid],
                                       std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    decode_queue.terminate(utils::AsyncQueueTerminateFast::No);

    for (size_t tid = 0; tid < std::size(worker_return_vals); ++tid) {
        const WorkerReturnStatus& rv = worker_return_vals[tid];
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            throw std::runtime_error{"(infer-samples) " + rv.message};
        } else {
            spdlog::warn("(infer-samples) " + rv.message);
        }
    }

    spdlog::debug("[infer_samples_in_parallel] Finished running inference.");
}

void decode_samples_in_parallel(std::vector<std::vector<secondary::ConsensusResult>>& results_cons,
                                std::vector<secondary::VariantCallingSample>& results_vc_data,
                                utils::AsyncQueue<DecodeData>& decode_queue,
                                secondary::Stats& stats,
                                std::atomic<bool>& worker_terminate,
                                polisher::WorkerReturnStatus& ret_status,
                                const secondary::DecoderBase& decoder,
                                const int32_t num_threads,
                                const int32_t min_depth,
                                const bool collect_vc_data,
                                const bool continue_on_exception) {
    utils::ScopedProfileRange spr1("decode_samples_in_parallel", 2);

    auto batch_decode = [&decoder, &stats, min_depth](const DecodeData& item, const int32_t tid) {
        utils::ScopedProfileRange spr2("decode_samples_in_parallel-batch_decode", 3);

        timer::TimerHighRes timer_total;
        timer::TimerHighRes timer_decode;

        // Decode output to bases and qualities.
        // Decode output to bases and qualities. Dimensions: [samples x haplotypes].
        std::vector<std::vector<secondary::ConsensusResult>> local_results =
                decoder.decode_bases(item.logits);

        // Dimensions: [samples x haplotypes].
        std::vector<std::vector<secondary::ConsensusResult>> final_results;
        final_results.reserve(std::size(local_results));

        const int64_t time_decode = timer_decode.GetElapsedMilliseconds();

        assert(std::size(local_results) == std::size(item.samples));
        assert(std::size(local_results) == std::size(item.trims));

        // Trim the overlapping sequences.
        timer::TimerHighRes timer_trim;
        for (int64_t j = 0; j < dorado::ssize(local_results); ++j) {
            // Empty local results should not be possible, but better be safe.
            if (std::empty(local_results[j])) {
                continue;
            }

            const secondary::Sample& sample = item.samples[j];
            const secondary::TrimInfo& trim = item.trims[j];
            const int64_t num_positions = dorado::ssize(sample.positions_major);

            if ((trim.start < 0) || (trim.start >= num_positions) || (trim.end <= 0) ||
                (trim.end > num_positions)) {
                spdlog::debug(
                        "Trim coordinate is < 0. j = {}, trim.start = {}, trim.end = {}, "
                        "trim.heuristic = {}, num_positions = {}",
                        j, trim.start, trim.end, trim.heuristic, num_positions);
                continue;
            }

            std::vector<secondary::Interval> good_intervals{
                    secondary::Interval{0, static_cast<int32_t>(num_positions)}};

            // Break on low coverage.
            if (min_depth > 0) {
                good_intervals.clear();

                const std::span<int64_t> depth(sample.depth.data_ptr<int64_t>(),
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

            int64_t draft_span = 0;

            if (std::size(good_intervals) == 1) {
                const int64_t draft_start = sample.positions_major[trim.start];
                const int64_t draft_end = sample.positions_major[trim.end - 1] + 1;
                draft_span = draft_end - draft_start;

                // Trim and mark the region.
                for (auto& result : local_results[j]) {
                    result.draft_id = sample.seq_id;
                    result.draft_start = draft_start;
                    result.draft_end = draft_end;
                    result.seq = result.seq.substr(trim.start, trim.end - trim.start);
                    result.quals = result.quals.substr(trim.start, trim.end - trim.start);
                }

                final_results.emplace_back(std::move(local_results[j]));

            } else {
                for (const auto& interval : good_intervals) {
                    if ((interval.start < 0) || (interval.end <= 0)) {
                        continue;
                    }

                    const int32_t start =
                            std::max(static_cast<int32_t>(trim.start), interval.start);
                    const int32_t end = std::min(static_cast<int32_t>(trim.end), interval.end);

                    if (end <= start) {
                        continue;
                    }

                    const int64_t draft_start = sample.positions_major[start];
                    const int64_t draft_end = sample.positions_major[end - 1] + 1;
                    draft_span = draft_end - draft_start;

                    std::vector<secondary::ConsensusResult> new_results;

                    for (auto& result : local_results[j]) {
                        secondary::ConsensusResult new_result;
                        new_result.draft_id = sample.seq_id;
                        new_result.draft_start = draft_start;
                        new_result.draft_end = draft_end;
                        new_result.seq = result.seq.substr(start, end - start);
                        new_result.quals = result.quals.substr(start, end - start);
                        new_results.emplace_back(std::move(new_result));
                    }

                    final_results.emplace_back(std::move(new_results));
                }
            }

            stats.add("processed", static_cast<double>(draft_span));
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
                            std::vector<std::vector<secondary::ConsensusResult>>& thread_results,
                            std::vector<secondary::VariantCallingSample>& thread_vc_data,
                            WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("decode_samples_in_parallel-worker", 3);
        at::InferenceMode infer_guard;

        while (!worker_terminate) {
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
                std::vector<std::vector<secondary::ConsensusResult>> results_samples =
                        batch_decode(item, tid);

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
                if (!continue_on_exception) {
                    ret_val = {.exception_thrown = true,
                               .message = std::string("Caught exception while decoding samples: '" +
                                                      std::string(e.what()) + "'")};
                    worker_terminate = true;
                    return;
                }

                spdlog::warn(
                        "Caught an exception when decoding a batch of samples. Skipping this "
                        "batch. Exception: {}",
                        e.what());
            }
        }
    };

    // Dimensions: threads x samples x haplotypes.
    std::vector<std::vector<std::vector<secondary::ConsensusResult>>> thread_results(num_threads);
    std::vector<std::vector<secondary::VariantCallingSample>> thread_vc_data(num_threads);

    cxxpool::thread_pool pool{static_cast<size_t>(num_threads)};

    std::vector<WorkerReturnStatus> worker_return_vals(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(thread_results[tid]),
                                       std::ref(thread_vc_data[tid]),
                                       std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    for (size_t tid = 0; tid < std::size(worker_return_vals); ++tid) {
        const WorkerReturnStatus& rv = worker_return_vals[tid];
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            worker_terminate = true;
            return;
        } else {
            spdlog::warn(rv.message);
        }
    }

    // Flatten the results.
    {
        size_t total_size = 0;
        for (const auto& vals : thread_results) {
            total_size += std::size(vals);
        }
        results_cons.clear();
        results_cons.reserve(total_size);

        // Take only the first haplotype of the consensus.
        for (size_t thread_id = 0; thread_id < std::size(thread_results); ++thread_id) {
            for (size_t sample_id = 0; sample_id < std::size(thread_results[thread_id]);
                 ++sample_id) {
                if (std::empty(thread_results[thread_id][sample_id])) {
                    continue;
                }
                results_cons.emplace_back(std::move(thread_results[thread_id][sample_id]));
            }
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

std::vector<std::vector<std::vector<secondary::ConsensusResult>>> construct_consensus_seqs(
        const secondary::Interval& region_batch,
        const std::vector<std::vector<secondary::ConsensusResult>>& all_results_cons,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const bool fill_gaps,
        const std::optional<char>& fill_char,
        hts_io::FastxRandomReader& draft_reader) {
    utils::ScopedProfileRange spr1("construct_consensus_seqs", 3);

    // Group samples by sequence ID.
    std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(region_batch.length());
    for (int32_t i = 0; i < dorado::ssize(all_results_cons); ++i) {
        const std::vector<secondary::ConsensusResult>& hap_results = all_results_cons[i];

        if (std::empty(hap_results)) {
            continue;
        }

        const int32_t draft_id = hap_results.front().draft_id;
        const int32_t local_id = draft_id - region_batch.start;

        // Skip filtered samples.
        if (draft_id < 0) {
            continue;
        }

        if ((draft_id >= dorado::ssize(draft_lens)) || (local_id < 0) ||
            (local_id >= dorado::ssize(groups))) {
            spdlog::error(
                    "Draft ID out of bounds! r.draft_id = {}, draft_lens.size = {}, "
                    "groups.size = {}",
                    draft_id, std::size(draft_lens), std::size(groups));
            continue;
        }
        groups[local_id].emplace_back(hap_results.front().draft_start, i);
    }

    // Dimensions: [draft_id x part_id x haplotype_id].
    std::vector<std::vector<std::vector<secondary::ConsensusResult>>> ret;

    // Consensus sequence - stitch the windows and write output.
    for (int64_t group_id = 0; group_id < dorado::ssize(groups); ++group_id) {
        const int64_t seq_id = group_id + region_batch.start;

        std::vector<std::pair<int64_t, int32_t>>& group = groups[group_id];
        std::sort(std::begin(group), std::end(group));  // Sort by start pos.

        const std::string& header = draft_lens[seq_id].first;

        // Dimensions: [part_id x haplotype_id].
        std::vector<std::vector<secondary::ConsensusResult>> consensus = stitch_sequence(
                draft_reader, header, all_results_cons, group, fill_gaps, fill_char);

        ret.emplace_back(std::move(consensus));
    }

    return ret;
}

std::vector<secondary::Variant> call_variants(
        std::atomic<bool>& worker_terminate,
        secondary::Stats& stats,
        const secondary::Interval& region_batch,
        const std::vector<secondary::VariantCallingSample>& vc_input_data,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& draft_readers,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        const float pass_min_qual,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t num_threads,
        const bool continue_on_exception) {
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
                            secondary::Stats& ps, WorkerReturnStatus& ret_val) {
        if ((start < 0) || (start >= end) || (end > dorado::ssize(results))) {
            throw std::runtime_error("Worker group_id is out of bounds! start = " +
                                     std::to_string(start) + ", end = " + std::to_string(end) +
                                     ", results.size = " + std::to_string(std::size(results)));
        }

        for (int32_t group_id = start; group_id < end; ++group_id) {
            if (worker_terminate) {
                return;
            }

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
                const auto joined_samples = join_samples(trimmed_vc_samples, draft, decoder);

                for (const auto& vc_sample : joined_samples) {
                    std::vector<secondary::Variant> variants = secondary::general_decode_variants(
                            decoder, vc_sample.seq_id, vc_sample.positions_major,
                            vc_sample.positions_minor, vc_sample.logits, draft, pass_min_qual,
                            ambig_ref, gvcf, true, true, true);

                    ps.add("processed", static_cast<double>(vc_sample.end() - vc_sample.start()));

                    results[group_id].insert(std::end(results[group_id]),
                                             std::make_move_iterator(std::begin(variants)),
                                             std::make_move_iterator(std::end(variants)));
                }
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Caught an exception in the call_variants::worker (tid = " << tid
                    << ", group_id = " << group_id << ", seq_id = " << seq_id << ", header = '"
                    << header
                    << "'). Not returning any variants for this group. Original message: '"
                    << e.what() << "'";

                results[group_id].clear();

                if (!continue_on_exception) {
                    ret_val = {.exception_thrown = true, .message = oss.str()};
                    worker_terminate = true;
                    return;
                }

                spdlog::warn(oss.str());
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

    std::vector<WorkerReturnStatus> worker_return_vals(std::size(thread_chunks));

#ifdef DEBUG_VC_DATA
    {
        for (int64_t ii = 0; ii < dorado::ssize(vc_input_data); ++ii) {
            const secondary::VariantCallingSample& vc_sample = vc_input_data[ii];
            const std::string& header = draft_lens[vc_sample.seq_id].first;
            const std::string draft = draft_readers[0]->fetch_seq(header);

            // Get raw probability data.
            const size_t batch_size = 1;
            const size_t seq_len = std::size(vc_sample.positions_major);
            const size_t num_haplotypes = 2;  // static_cast<size_t>(probs_3D.size(1));
            const size_t num_classes = std::size(decoder.get_label_scheme_symbols());
            const dorado::Span<const float> raw_probs_data(
                    vc_sample.logits.data_ptr<float>(),
                    batch_size * seq_len * num_haplotypes * num_classes);

            // Consensus sequences.
            const std::vector<std::vector<secondary::ConsensusResult>> cons_seqs_with_gaps_all =
                    dorado::secondary::decode_batch_bases_impl(decoder.get_label_scheme_symbols(),
                                                               raw_probs_data, batch_size, seq_len,
                                                               num_haplotypes, num_classes);

            std::cout << "Debugging data before merging. vc_sample.logits.shape = ["
                      << utils::tensor_shape_as_string(vc_sample.logits) << "]\n";
            std::cout << "batch_size = " << batch_size << "\n";
            std::cout << "seq_len = " << seq_len << "\n";
            std::cout << "num_haplotypes = " << num_haplotypes << "\n";
            std::cout << "num_classes = " << num_classes << "\n";
            std::vector<std::string_view> cons_view;
            for (const secondary::ConsensusResult& val : cons_seqs_with_gaps_all.front()) {
                cons_view.emplace_back(val.seq);
            }
            std::cout << "vc_sample.seq_id = " << vc_sample.seq_id << '\n';
            const std::string ref_seq_with_gaps = dorado::secondary::extract_draft_with_gaps(
                    draft, vc_sample.positions_major, vc_sample.positions_minor);
            dorado::secondary::print_slice(
                    std::cout, ref_seq_with_gaps, cons_view, vc_sample.positions_major,
                    vc_sample.positions_minor,
                    std::vector<bool>(std::size(vc_sample.positions_major), false), 0, -1, 0, -1);
            std::cout << "vc_sample.logits tensor =\n" << vc_sample.logits << "\n";
        }
    }
#endif

    // Add worker tasks.
    for (int32_t tid = 0; tid < static_cast<int32_t>(std::size(thread_chunks)); ++tid) {
        const auto [chunk_start, chunk_end] = thread_chunks[tid];
        futures.emplace_back(pool.push(worker, tid, chunk_start, chunk_end,
                                       std::ref(thread_results), std::ref(stats),
                                       std::ref(worker_return_vals[tid])));
    }

    // Join and catch exceptions.
    for (auto& f : futures) {
        f.get();
    }

    for (size_t tid = 0; tid < std::size(worker_return_vals); ++tid) {
        const WorkerReturnStatus& rv = worker_return_vals[tid];
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            throw std::runtime_error{"(call variants) " + rv.message};
        } else {
            spdlog::warn("(call variants) " + rv.message);
        }
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

secondary::ModelConfig load_model(const std::filesystem::path& model_dir,
                                  const bool load_scripted_model) {
    const std::filesystem::path model_config_file = model_dir / "config.toml";
    const std::string model_file = load_scripted_model ? "model.pt" : "weights.pt";

    spdlog::info("Parsing the model config: {}", (model_config_file).string());
    secondary::ModelConfig model_config =
            secondary::parse_model_config(model_config_file, model_file);
    return model_config;
}

}  // namespace dorado::polisher
