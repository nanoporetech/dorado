#include "variant/variant_impl.h"

#include "hts_utils/FastxRandomReader.h"
#include "secondary/architectures/model_factory.h"
#include "secondary/common/batching.h"
#include "secondary/common/region.h"
#include "secondary/consensus/sample_collate_utils.h"
#include "secondary/consensus/variant_calling.h"
#include "secondary/features/encoder_utils.h"
#include "torch_utils/gpu_profiling.h"
#include "torch_utils/tensor_utils.h"
#include "utils/container_utils.h"
#include "utils/memory_utils.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/timer_high_res.h"

#include <ATen/ATen.h>
#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

// #define DEBUG_INFERENCE_DATA
// #define DEBUG_DUMP_INFERENCE_TENSORS_TO_DISK
constexpr bool DEBUG_VC_SAMPLES = false;

#ifdef NDEBUG
#define LOG_TRACE(...)
#else
#define LOG_TRACE(...) spdlog::trace(__VA_ARGS__)
#endif

namespace dorado::smallvar {

namespace {

std::vector<secondary::DeviceInfo> init_devices(const std::string& devices_str) {
    std::vector<secondary::DeviceInfo> devices;

    if (devices_str == "cpu") {
        torch::Device torch_device = torch::Device(devices_str);
        devices.emplace_back(
                secondary::DeviceInfo{.name = devices_str,
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
            devices.emplace_back(secondary::DeviceInfo{.name = val,
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

bool sample_has_inference_features(const secondary::Sample& sample) {
    return !std::empty(sample.positions_major) && sample.features.defined() &&
           (sample.features.numel() > 0);
}

bool variant_is_in_unique_bam_window_span(const secondary::Variant& variant,
                                          const secondary::Window& bam_window) {
    return (variant.pos >= bam_window.start_no_overlap) &&
           (variant.pos < bam_window.end_no_overlap);
}

}  // namespace

void signal_worker_terminate(std::atomic<bool>& worker_terminate) {
    worker_terminate.store(true, std::memory_order_release);
    worker_terminate.notify_all();
}

VariantResources create_resources(const secondary::ModelConfig& model_config,
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
                                  const secondary::KadayashiOptions& kadayashi_opt,
                                  const bool legacy_feature_gen) {
    VariantResources resources;

    spdlog::info("Initializing the devices.");
    resources.devices = init_devices(device_str);
    if (std::empty(resources.devices)) {
        throw std::runtime_error("Zero devices initialized! Need at least one device to run.");
    }

    spdlog::debug("Initialized devices:");
    for (std::size_t device_id = 0; device_id < std::size(resources.devices); ++device_id) {
        const secondary::DeviceInfo& dev_info = resources.devices[device_id];
        spdlog::debug("    - [device_id = {}] name = {}, available_memory = {:.2f} GB", device_id,
                      dev_info.name, dev_info.available_memory_GB);
    }

    // Construct the model.
    spdlog::debug("[create_resources] Loading the model.");
    const auto create_models = [&]() {
        std::vector<std::shared_ptr<secondary::ModelTorchBase>> ret;
        std::vector<c10::optional<c10::Stream>> ret_streams;

        for (std::size_t device_id = 0; device_id < std::size(resources.devices); ++device_id) {
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
                if (device_info.device.is_cuda() && !full_precision) {
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
        resources.encoders.emplace_back(encoder_factory(
                model_config, in_ref_fn, in_aln_bam_fn, read_group, tag_name, tag_value, true,
                min_snp_accuracy, tag_keep_missing_override, min_mapq_override, haptag_source,
                phasing_bin_fn, kadayashi_opt, legacy_feature_gen));
    }

    spdlog::info("Creating the decoder.");
    resources.decoder = decoder_factory(model_config);

    return resources;
}

std::vector<secondary::Variant> convert_variants(
        const std::vector<kadayashi::variant_dorado_style_t>& kadayashi_variants,
        const int32_t seq_id,
        const int32_t ploidy,
        const float pass_min_qual) {
    std::vector<secondary::Variant> ret;
    ret.reserve(std::ssize(kadayashi_variants));

    for (int64_t j = 0; j < std::ssize(kadayashi_variants); ++j) {
        const kadayashi::variant_dorado_style_t& var = kadayashi_variants[j];

        secondary::Variant new_var = secondary::Variant{
                .seq_id = seq_id,
                .pos = static_cast<int64_t>(var.pos),
                .ref = var.ref,
                .alts = var.alts,
                .filter = "",
                .info = {},
                .qual = static_cast<float>(var.qual),
                .genotype = {},
                .rstart = 0,
                .rend = 0,
        };

        // Kadayashi does not output an alt for every haplotype.
        // Alts which match the reference are added here.
        while (!std::empty(new_var.alts) && (std::ssize(new_var.alts) < ploidy)) {
            if (var.genotype.first != var.genotype.second) {
                // The alt matches the ref (het variant).
                new_var.alts.emplace_back(new_var.ref);
            } else {
                // Hom variant.
                new_var.alts.emplace_back(new_var.alts.front());
            }
        }

        new_var = secondary::normalize_genotype(new_var, ploidy, pass_min_qual);

        ret.emplace_back(std::move(new_var));
    }

    return ret;
}

namespace {
secondary::IntervalTreeInt64 build_interval_tree(const std::vector<int64_t>& in_candidate_sites) {
    std::vector<interval_tree::Interval<int64_t, int64_t>> intervals;
    intervals.reserve(std::size(in_candidate_sites));
    for (const int64_t pos : in_candidate_sites) {
        intervals.emplace_back(pos, pos + 1, 0);
    }
    return secondary::IntervalTreeInt64(std::move(intervals));
}

std::pair<std::vector<secondary::Sample>, std::vector<secondary::Variant>>
process_single_bam_window(
        const secondary::Window& bam_window,
        secondary::EncoderBase& encoder,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        [[maybe_unused]] const std::vector<std::string>& draft_seqs,
        const secondary::VariantCandidateSource candidate_source,
        const std::optional<secondary::IntervalTreesInt64Map>& candidate_trees_from_file,
        const int32_t ploidy,
        const float pass_min_qual,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t variant_flanking_bases,
        const bool tiled_regions,
        const bool tiled_ext_flanks,
        const int64_t tiled_ext_major,
        const int64_t tiled_ext_min_cov,
        const float tiled_ext_cov_fract,
        const int32_t min_depth,
        const bool model_requires_draft,
        const int32_t tid) {
    if ((bam_window.seq_id < 0) || (bam_window.seq_id >= std::ssize(draft_lens)) ||
        (bam_window.seq_id >= std::ssize(draft_seqs))) {
        throw std::runtime_error{
                fmt::format("bam_window.seq_id ({}) is out of bounds for draft_lens (size = {}) or "
                            "the number of loaded reference sequences (size = {})",
                            bam_window.seq_id, std::size(draft_lens), std::size(draft_seqs))};
    }

    const std::string& ref_name = draft_lens[bam_window.seq_id].first;

    spdlog::trace(
            "[process_single_bam_window tid = {}] Starting to generate the sample for region: "
            "{}:{}-{}",
            tid, ref_name, (bam_window.start + 1), bam_window.end);

    kadayashi::varcall_result_t kadayashi_result =
            encoder.produce_haplotags(ref_name, bam_window.start, bam_window.end);

    // Increment the haplotag from 0/1 -> 1/2 because the model was trained on that.
    for (auto& [key, val] : kadayashi_result.qname2hp) {
        ++val;
    }

    // Optionally convert variants only if the source is COMPUTE.
    std::vector<secondary::Variant> pass_variants;
    std::vector<int64_t> candidate_sites;

    if (candidate_source == secondary::VariantCandidateSource::COMPUTE) {
        // Convert all variants.
        std::vector<secondary::Variant> all_variants = convert_variants(
                kadayashi_result.variants, bam_window.seq_id, ploidy, pass_min_qual);

        // Storage space for pass/candidate variants.
        pass_variants.reserve(std::size(all_variants));
        candidate_sites.reserve(std::size(all_variants));

        // Extract the set of candidate variant sites (low-qual variants) and PASS variants.
        for (secondary::Variant& var : all_variants) {
            if (!variant_is_in_unique_bam_window_span(var, bam_window)) {
                continue;
            }

            if (var.qual < pass_min_qual) {
                candidate_sites.emplace_back(var.pos);
            } else {
                pass_variants.emplace_back(std::move(var));
            }
        }
    }

    spdlog::trace(
            "[process_single_bam_window tid = {}] Done haplotagging and calling simple variants "
            "for region: {}:{}-{}",
            tid, ref_name, (bam_window.start + 1), bam_window.end);

    spdlog::trace(
            "[process_single_bam_window tid = {}] About to generate the feature matrix for region: "
            "{}:{}-{}",
            tid, ref_name, (bam_window.start + 1), bam_window.end);

    secondary::Sample sample = encoder.encode_region(ref_name, bam_window.start, bam_window.end,
                                                     bam_window.seq_id, kadayashi_result.qname2hp);

    spdlog::debug(
            "[process_single_bam_window tid = {}] Generated sample for region: {}:{}-{}, sample: "
            "[{}]",
            tid, ref_name, (bam_window.start + 1), bam_window.end,
            secondary::sample_to_string(sample));

    if (std::empty(sample.positions_major)) {
        return std::pair<std::vector<secondary::Sample>, std::vector<secondary::Variant>>(
                std::vector<secondary::Sample>{}, std::move(pass_variants));
    }

    // Split on coordinate gaps and exclude low-coverage columns before inference.
    const std::vector<secondary::Interval64> intervals =
            secondary::find_sample_intervals(sample, true, min_depth);

    // Non-const because it will be moved below.
    std::vector<secondary::Sample> local_samples =
            secondary::split_sample_on_intervals(sample, intervals);

    constexpr int64_t MIN_SAMPLE_POSITIONS_FOR_VARIANT_INFERENCE = 5;
    std::erase_if(local_samples, [](const secondary::Sample& local_sample) {
        return std::ssize(local_sample.positions_major) <
               MIN_SAMPLE_POSITIONS_FOR_VARIANT_INFERENCE;
    });

    spdlog::trace(
            "[process_single_bam_window tid = {}] After splitting on min-depth/discontinuities: "
            "{}:{}-{}, "
            "local_samples.size = {}",
            tid, ref_name, (bam_window.start + 1), bam_window.end, std::size(local_samples));

    // Candidate variants, if needed.
    secondary::IntervalTreesInt64Map candidate_trees;
    if (candidate_trees_from_file &&
        (candidate_source == secondary::VariantCandidateSource::FILE)) {
        // // There are no simple variants to merge in this case, merging should be handled outside.
        spdlog::debug("Using candidate sites from an input file.");

        const auto it = candidate_trees_from_file->find(bam_window.seq_id);
        if (it == std::cend(*candidate_trees_from_file)) {
            return {};
        }

        // TODO: This makes a copy. Consider a more efficient way.
        candidate_trees = {
                {bam_window.seq_id, it->second},
        };

    } else if (candidate_source == secondary::VariantCandidateSource::COMPUTE) {
        candidate_trees = {
                {bam_window.seq_id, build_interval_tree(candidate_sites)},
        };
    }

    // Optionally run splitting
    if (candidate_source != secondary::VariantCandidateSource::NONE) {
        if (!tiled_regions) {
            local_samples = split_samples_around_positions(
                    std::move(local_samples), candidate_trees, window_len, variant_flanking_bases);
        } else {
            local_samples = split_samples_tiled_with_candidates(
                    std::move(local_samples), candidate_trees, window_len, window_overlap,
                    tiled_ext_flanks, tiled_ext_major, tiled_ext_min_cov, tiled_ext_cov_fract);
        }
    } else {
        local_samples = split_samples(std::move(local_samples), window_len, window_overlap);
    }

    if (model_requires_draft) {
        const std::string_view draft_seq(draft_seqs[bam_window.seq_id]);
        for (auto& local_sample : local_samples) {
            local_sample.draft_seq = {secondary::draft_encoding_from_seq(
                    local_sample.positions_major, local_sample.positions_minor, draft_seq)};
        }
    }

    spdlog::trace(
            "[process_single_bam_window tid = {}] After final sample extraction: {}:{}-{}, "
            "local_samples.size = {}",
            tid, ref_name, (bam_window.start + 1), bam_window.end, std::size(local_samples));

    return std::pair<std::vector<secondary::Sample>, std::vector<secondary::Variant>>(
            std::move(local_samples), std::move(pass_variants));
}
}  // namespace

void worker_sample_producer(
        utils::AsyncQueue<secondary::Window>& input_queue,
        utils::AsyncQueue<InferenceData>& output_queue,
        std::vector<ChromosomeReduceData>& chrom_reduce_data,
        VariantResources& resources,
        secondary::Stats& stats,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        const std::vector<std::vector<secondary::Window>>& bam_regions,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<std::string>& draft_seqs,
        const secondary::VariantCandidateSource candidate_source,
        const std::optional<secondary::IntervalTreesInt64Map>& candidate_trees_from_file,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap,
        const int32_t variant_flanking_bases,
        const bool continue_on_exception,
        const int32_t ploidy,
        const float pass_min_qual,
        const bool tiled_regions,
        const bool tiled_ext_flanks,
        const int64_t tiled_ext_major,
        const int64_t tiled_ext_min_cov,
        const float tiled_ext_cov_fract,
        const int32_t min_depth) {
    utils::ScopedProfileRange spr1("sample_producer", 2);

    if (std::size(draft_lens) != std::size(draft_seqs)) {
        throw std::runtime_error{
                "Number of loaded reference sequence lengths and sequences differs. draft_lens = " +
                std::to_string(std::size(draft_lens)) +
                ", draft_seqs = " + std::to_string(std::size(draft_seqs))};
    }

    const auto worker = [&](const int32_t tid, secondary::WorkerReturnStatus& ret_val) {
        while (!worker_terminate) {
            utils::ScopedProfileRange spr3("sample_producer-worker-while", 4);

            spdlog::trace(
                    "[sample_producer::worker {}] Waiting to pop a BAM region. Queue size: {}", tid,
                    std::size(input_queue));

            secondary::Window bam_window;
            const auto pop_status = input_queue.try_pop(bam_window);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            std::ostringstream oss;
            oss << bam_window;
            spdlog::trace("[sample_producer::worker {}] Popped BAM window: [{}]", tid, oss.str());

            bool reduce_data_updated = false;
            try {
                auto [local_samples, simple_variants] = process_single_bam_window(
                        bam_window, *resources.encoders[tid], draft_lens, draft_seqs,
                        candidate_source, candidate_trees_from_file, ploidy, pass_min_qual,
                        window_len, window_overlap, variant_flanking_bases, tiled_regions,
                        tiled_ext_flanks, tiled_ext_major, tiled_ext_min_cov, tiled_ext_cov_fract,
                        min_depth, resources.models[0]->requires_ref(), tid);
                stats.add("processed",
                          static_cast<double>(std::max<int64_t>(
                                  0, bam_window.end_no_overlap - bam_window.start_no_overlap)));
                const int64_t num_outgoing_samples =
                        std::count_if(std::cbegin(local_samples), std::cend(local_samples),
                                      sample_has_inference_features);

                // Update the info needed to reduce the results.
                {
                    ChromosomeReduceData& rd = chrom_reduce_data[bam_window.seq_id];

                    std::lock_guard<std::mutex> lock(rd.mtx);

                    --rd.remaining_bam_regions;
                    if (rd.remaining_bam_regions < 0) {
                        throw std::runtime_error{fmt::format(
                                "chrom_reduce_data[{}].remaining_bam_regions ({}) became "
                                "negative while processing bam_window {}.",
                                bam_window.seq_id, rd.remaining_bam_regions,
                                secondary::window_to_string(bam_window))};
                    }
                    rd.num_samples += num_outgoing_samples;

                    spdlog::trace(
                            "[sample_producer::worker {}] bam_window = [{}], rd.ready = {}, "
                            "rd.num_bam_regions = {}, "
                            "rd.remaining_bam_regions = "
                            "{}, rd.num_samples = {}, bam_regions[bam_window.seq_id].size = {}",
                            tid, secondary::window_to_string(bam_window), rd.ready,
                            rd.num_bam_regions, rd.remaining_bam_regions, rd.num_samples,
                            std::size(bam_regions[bam_window.seq_id]));

                    if (rd.remaining_bam_regions == 0) {
                        rd.ready = true;
                    }
                    reduce_data_updated = true;

                    // Add the simple variants to the results for this chromosome.
                    rd.variants_simple.reserve(std::size(rd.variants_simple) +
                                               std::size(simple_variants));
                    rd.variants_simple.insert(std::end(rd.variants_simple),
                                              std::make_move_iterator(std::begin(simple_variants)),
                                              std::make_move_iterator(std::end(simple_variants)));
                }

                // Create single-element InferenceData items of samples.
                for (int64_t i = 0; i < std::ssize(local_samples); ++i) {
                    auto& sample = local_samples[i];
                    if (!sample_has_inference_features(sample)) {
                        continue;
                    }
                    InferenceData item{
                            .samples = {std::move(sample)},
                    };
                    output_queue.try_push(std::move(item));
                }
            } catch (const std::exception& e) {
                if (!continue_on_exception) {
                    ret_val = {
                            .exception_thrown = true,
                            .message = std::string("Caught exception while producing samples: '" +
                                                   std::string(e.what()) + "'")};
                    signal_worker_terminate(worker_terminate);
                    return;
                }

                if (!reduce_data_updated) {
                    ChromosomeReduceData& rd = chrom_reduce_data[bam_window.seq_id];
                    std::lock_guard<std::mutex> lock(rd.mtx);
                    --rd.remaining_bam_regions;
                    if (rd.remaining_bam_regions == 0) {
                        rd.ready = true;
                    }
                }

                spdlog::warn(
                        "Caught exception while producing samples. Skipping this BAM window. "
                        "Exception: {}",
                        e.what());
            }
        }
    };

    // Create the thread pool, futures and results.
    const std::size_t actual_threads =
            std::min(num_threads, static_cast<int32_t>(std::size(resources.encoders)));
    cxxpool::thread_pool pool{actual_threads};

    std::vector<std::future<void>> futures;
    futures.reserve(actual_threads);

    // std::vector<secondary::Sample> results(std::size(windows));
    std::vector<secondary::WorkerReturnStatus> worker_return_vals(actual_threads);

    // Add jobs to the pool.
    for (int32_t tid = 0; tid < static_cast<int32_t>(actual_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    if (!worker_terminate.load(std::memory_order_acquire)) {
        output_queue.terminate(utils::AsyncQueueTerminateFast::No);
    }

    for (const secondary::WorkerReturnStatus& rv : worker_return_vals) {
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            signal_worker_terminate(worker_terminate);
            return;
        } else {
            spdlog::warn(rv.message);
        }
    }
}

void worker_batch_producer(utils::AsyncQueue<InferenceData>& input_queue,
                           utils::AsyncQueue<InferenceData>& output_queue,
                           std::atomic<bool>& worker_terminate,
                           secondary::WorkerReturnStatus& ret_status,
                           const dorado::secondary::ModelTorchBase& model,
                           const int32_t window_len,
                           const int32_t batch_size,
                           const double max_available_mem,
                           const bool continue_on_exception) {
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
        const int64_t num_samples = std::ssize(buffer.samples);
        const int64_t new_batch_size = (num_samples < 8) ? num_samples : (8 * (num_samples / 8));

        // Get the first batch_size elements and push them to the queue.
        InferenceData new_buffer;
        for (int64_t i = 0; i < new_batch_size; ++i) {
            new_buffer.samples.emplace_back(std::move(buffer.samples[i]));
        }
        queue.try_push(std::move(new_buffer));

        // Get the remaining items and update the buffer.
        InferenceData remainder;
        for (int64_t i = new_batch_size; i < std::ssize(buffer.samples); ++i) {
            remainder.samples.emplace_back(std::move(buffer.samples[i]));
        }
        buffer.samples = std::move(remainder.samples);
    };

    InferenceData buffer;

    while (!worker_terminate) {
        try {
            utils::ScopedProfileRange spr3("batch_producer-while", 4);

            LOG_TRACE(
                    "[batch_producer] Waiting to pop a sample from the input_queue. Queue size: "
                    "{}",
                    std::size(input_queue));

            // Pop an element.
            InferenceData item;
            const auto pop_status = input_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            LOG_TRACE("[batch_producer] Popped an item of {} samples from the input queue.",
                      std::ssize(item.samples));

            for (int64_t i = 0; i < std::ssize(item.samples); ++i) {
                auto& sample = item.samples[i];

                if (!sample_has_inference_features(sample)) {
                    spdlog::trace(
                            "[batch_producer] Skipping sample without inferable feature rows. "
                            "Sample: [{}].",
                            secondary::sample_to_string(sample));
                    continue;
                }

                // Odd samples will be processed alone.
                if (std::ssize(sample.positions_major) != window_len) {
                    InferenceData remainder_buffer;
                    remainder_buffer.samples.emplace_back(std::move(sample));
                    spdlog::trace(
                            "[batch_producer] Pushing a remainder batch of data to infer_data "
                            "queue. Sample: [{}].",
                            secondary::sample_to_string(sample));
                    output_queue.try_push(std::move(remainder_buffer));
                    continue;
                }

                const std::vector<int64_t> curr_batch_shape =
                        secondary::compute_collated_padded_shape(buffer.samples);

                const double estimated_memory_curr =
                        std::empty(curr_batch_shape)
                                ? 0.0
                                : model.estimate_batch_memory(curr_batch_shape);

                // Cut batches either on memory consumption or on the absolute count.
                if (batch_size <= 0) {
                    // Auto batch size computation.
                    const std::vector<int64_t> next_batch_shape =
                            secondary::compute_collated_padded_shape(buffer.samples, sample);

                    const double estimated_memory_next =
                            model.estimate_batch_memory(next_batch_shape);

                    if (!std::empty(buffer.samples) &&
                        (estimated_memory_next >= max_available_mem)) {
                        spdlog::trace(
                                "[batch_producer] Estimating batch memory for auto batch size:");
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
                                "[batch_producer] Pushing a batch of data to infer_data queue "
                                "(1a). "
                                "buffer.samples.size() = {}",
                                std::size(buffer.samples));

                        move_buffer_data_to_queue(buffer, output_queue, false);
                    }

                } else {
                    if (std::ssize(buffer.samples) >= batch_size) {
                        // Fixed batch size.
                        spdlog::trace(
                                "[batch_producer] Estimating batch memory for fixed batch "
                                "size:");
                        spdlog::trace("    - max_available_mem = {} GB", max_available_mem);
                        spdlog::trace("    - estimated_memory_curr = {} GB", estimated_memory_curr);
                        spdlog::trace(
                                "    - curr_batch_shape = {}",
                                utils::print_container_as_string(curr_batch_shape, ",", true));

                        spdlog::trace(
                                "[batch_producer] Pushing a batch of data to infer_data queue "
                                "(1b). "
                                "buffer.samples.size() = {}",
                                std::size(buffer.samples));

                        move_buffer_data_to_queue(buffer, output_queue, true);
                    }
                }

                // Expand the current buffer.
                buffer.samples.emplace_back(std::move(sample));
            }
        } catch (const std::exception& e) {
            if (!continue_on_exception) {
                // Cannot throw because this is a worker function intended to run on a separate thread.
                // Instead, communicate the error and return.
                ret_status = {.exception_thrown = true, .message = e.what()};
                signal_worker_terminate(worker_terminate);
                return;
            }
            spdlog::warn("Caught exception in the batch producer. Skipping the current item.");
            continue;
        }
    }

    if (!std::empty(buffer.samples)) {
        const std::vector<int64_t> curr_batch_shape =
                secondary::compute_collated_padded_shape(buffer.samples);
        const double estimated_memory_curr =
                std::empty(curr_batch_shape) ? 0.0 : model.estimate_batch_memory(curr_batch_shape);

        spdlog::trace("[batch_producer] Estimating batch memory for the final batch:");
        spdlog::trace("    - max_available_mem = {} GB", max_available_mem);
        spdlog::trace("    - estimated_memory_curr = {} GB", estimated_memory_curr);
        spdlog::trace("    - curr_batch_shape = {}",
                      utils::print_container_as_string(curr_batch_shape, ",", true));

        spdlog::trace(
                "[batch_producer] Pushing a batch of data to infer_data queue (2). "
                "buffer.samples.size() = {} (final)",
                std::size(buffer.samples));

        // First, push the N*8 elements to the queue for efficiency.
        move_buffer_data_to_queue(buffer, output_queue, false);

        // Next, push the remainder if the buffer is not empty.
        move_buffer_data_to_queue(buffer, output_queue, true);

        spdlog::debug("[batch_producer] Pushed final batch for inference to infer_data queue.");
    }

    if (!worker_terminate.load(std::memory_order_acquire)) {
        output_queue.terminate(utils::AsyncQueueTerminateFast::No);
    }
}

void worker_infer_samples_in_parallel(
        utils::AsyncQueue<InferenceData>& batch_queue,
        utils::AsyncQueue<DecodeData>& decode_queue,
        std::vector<std::shared_ptr<secondary::ModelTorchBase>>& models,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
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
            for (int64_t ii = 0; ii < std::ssize(batch.samples); ++ii) {
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
        dorado::secondary::BatchedData batched_data;
        int64_t time_collate = 0;
        int64_t time_move_to_device = 0;

        {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-collate", 4);
            timer::TimerHighRes timer_collate;
            std::vector<torch::Tensor> batch_features;
            batch_features.reserve(std::size(batch.samples));
            for (const auto& sample : batch.samples) {
                batch_features.emplace_back(sample.features);
            }
            const bool use_pinned_memory = (model.get_device().type() == torch::kCUDA);
            batched_data.features =
                    encoders[tid]->collate(std::move(batch_features), use_pinned_memory);

            spdlog::trace("In batching, model requires ref: {}", model.requires_ref());
            if (model.requires_ref()) {
                if (!batch.samples[0].draft_seq) {
                    throw std::runtime_error{
                            "Model requires reference but these are missing from samples."};
                }
                std::vector<torch::Tensor> refseqs;
                refseqs.reserve(std::ssize(batch.samples));
                for (const auto& sample : batch.samples) {
                    refseqs.emplace_back(sample.draft_seq->view({-1, 1, 1}));
                }
                batched_data.refseqs = {encoders[tid]
                                                ->collate(std::move(refseqs), use_pinned_memory)
                                                .view({std::ssize(batch.samples), -1})};
            }
            spdlog::trace("Post-batching, batch has refs: {}", batched_data.refseqs ? true : false);

            time_collate = timer_collate.GetElapsedMilliseconds();
        }

        {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-move_to_device", 4);
            timer::TimerHighRes timer_move_to_device;
            const bool non_blocking = (model.get_device().type() == torch::kCUDA);
            batched_data = model.prepare_batch_input(std::move(batched_data), non_blocking);
            time_move_to_device = timer_move_to_device.GetElapsedMilliseconds();
            spdlog::trace("Post-move, batch has refs: {}", batched_data.refseqs ? true : false);
        }

        const std::string input_batch_tensor_shape_str =
                utils::tensor_shape_as_string(batched_data.features);

        // Debug output.
        {
            const std::vector<int64_t> input_batch_tensor_shape =
                    batched_data.features.sizes().vec();
            const double estimated_memory = model.estimate_batch_memory(input_batch_tensor_shape);
            spdlog::debug(
                    "[consumer {}] Running inference: batch tensor shape = [{}], estimated memory "
                    "= {:.2f} GB",
                    tid, input_batch_tensor_shape_str, estimated_memory);
        }

        // Inference.
        torch::Tensor output;
        torch::Tensor output_on_device;
        int64_t time_forward = 0;
        int64_t time_move_to_host = 0;

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
                std::cout << "[infer] input: batched_data.features.shape = "
                          << utils::tensor_shape_as_string(batched_data.features) << "\n";
                std::cout << "[infer] input: batched_data.features =\n"
                          << batched_data.features << "\n";
                utils::save_tensor(batched_data.features, "debug.tensor.in.pt");
            }
#endif

            timer::TimerHighRes timer_forward;

            try {
                output_on_device = model.predict_on_device_batch(batched_data);
            } catch (const std::exception& e) {
                spdlog::error("Exception caught: {}", e.what());
                throw;
            }

            time_forward = timer_forward.GetElapsedMilliseconds();

#ifdef DEBUG_INFERENCE_DATA
            {
                std::cout << "[infer] output_device.shape = "
                          << utils::tensor_shape_as_string(output_on_device) << "\n";
                std::cout << "[infer] output_device =\n" << output_on_device << "\n";
                utils::save_tensor(output_on_device, "debug.tensor.out.pt");
            }
#endif
        }

        {
            utils::ScopedProfileRange spr3("infer_samples_in_parallel-move_to_host", 4);
            timer::TimerHighRes timer_move_to_host;
            output = output_on_device.cpu();
            time_move_to_host = timer_move_to_host.GetElapsedMilliseconds();
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
            const int64_t time_total = timer_total.GetElapsedMilliseconds();

            spdlog::trace(
                    "[consumer {}] Computed batch inference. Timings - collate: {} ms, "
                    "move_to_device: {} ms, forward: {} ms, move_to_host: {} ms, "
                    "total = {}, batched_data.features.shape = [{}]",
                    tid, time_collate, time_move_to_device, time_forward, time_move_to_host,
                    time_total, input_batch_tensor_shape_str);
        }

        return output;
    };

    const auto worker = [&](const int32_t tid, secondary::ModelTorchBase& model,
                            [[maybe_unused]] const c10::optional<c10::Stream>& stream,
                            secondary::WorkerReturnStatus& ret_val) {
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
                    signal_worker_terminate(worker_terminate);
                    return;
                }
            }
        }
    };

    if (std::size(models) > std::size(encoders)) {
        spdlog::warn(
                "There are more models than there are encoders! Num models: {}, num_encoders: {}. "
                "Using fewer models.",
                std::size(models), std::size(encoders));
    }

    const size_t num_threads = std::min(std::size(models), std::size(encoders));
    cxxpool::thread_pool pool{num_threads};

    std::vector<secondary::WorkerReturnStatus> worker_return_vals(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(*models[tid]), streams[tid],
                                       std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    if (!worker_terminate.load(std::memory_order_acquire)) {
        decode_queue.terminate(utils::AsyncQueueTerminateFast::No);
    }

    for (const secondary::WorkerReturnStatus& rv : worker_return_vals) {
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            signal_worker_terminate(worker_terminate);
            return;
        } else {
            spdlog::warn("(infer-samples) {}", rv.message);
        }
    }

    spdlog::debug("[infer_samples_in_parallel] Finished running inference.");
}

void worker_separate_decode_data(utils::AsyncQueue<DecodeData>& input_queue,
                                 utils::AsyncQueue<secondary::VariantCallingSample>& out_queue,
                                 std::atomic<bool>& worker_terminate,
                                 secondary::WorkerReturnStatus& ret_status,
                                 const int32_t num_threads,
                                 const bool continue_on_exception) {
    utils::ScopedProfileRange spr1("separate_decode_data", 2);

    const auto worker = [&](const int32_t tid, secondary::WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("separate_decode_data-worker", 3);
        at::InferenceMode infer_guard;

        while (!worker_terminate) {
            utils::ScopedProfileRange spr3("separate_decode_data-worker-while", 4);

            DecodeData item;
            const auto pop_status = input_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            try {
                const int64_t tensor_batch_size =
                        (item.logits.sizes().size() == 0) ? 0 : item.logits.size(0);

                spdlog::trace(
                        "[separate_decode_data {}] Popped data: item.logits.shape = {}, "
                        "tensor_batch_size = {}, queue size: {}",
                        tid, utils::tensor_shape_as_string(item.logits), tensor_batch_size,
                        std::size(input_queue));

                // This should handle the timeout case too.
                if (tensor_batch_size == 0) {
                    continue;
                }

                // Split data for each input sample in the batch.
                const std::vector<torch::Tensor> split_logits = item.logits.unbind(0);

                // Sanity check that the number of inferred samples matches the expected input.
                if (std::size(item.samples) != std::size(split_logits)) {
                    throw std::runtime_error(
                            "The batch size of results produced by the batch inference does not "
                            "match the expected batch size of the input tensor! Variant calling "
                            "for this batch will not be possible. samples.size = " +
                            std::to_string(std::size(item.samples)) +
                            ", split_logits.size = " + std::to_string(std::size(split_logits)));
                }

                // Create the variant calling data.
                spdlog::trace("[separate_decode_data {}] About to push {} items to vc_data_queue.",
                              tid, std::ssize(item.samples));
                for (int64_t i = 0; i < std::ssize(item.samples); ++i) {
                    out_queue.try_push(secondary::VariantCallingSample{
                            item.samples[i].seq_id, std::move(item.samples[i].positions_major),
                            std::move(item.samples[i].positions_minor), split_logits[i]});
                }
                spdlog::trace("[separate_decode_data {}] Pushed {} items to vc_data_queue.", tid,
                              std::ssize(item.samples));

            } catch (const std::exception& e) {
                if (!continue_on_exception) {
                    ret_val = {
                            .exception_thrown = true,
                            .message = std::string("Caught exception while separating samples: '" +
                                                   std::string(e.what()) + "'")};
                    signal_worker_terminate(worker_terminate);
                    return;
                }

                spdlog::warn(
                        "Caught an exception while separating samples. Skipping this "
                        "batch. Exception: {}",
                        e.what());
            }
        }
    };

    cxxpool::thread_pool pool{static_cast<size_t>(num_threads)};

    std::vector<secondary::WorkerReturnStatus> worker_return_vals(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    if (!worker_terminate.load(std::memory_order_acquire)) {
        out_queue.terminate(utils::AsyncQueueTerminateFast::No);
    }

    for (const secondary::WorkerReturnStatus& rv : worker_return_vals) {
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            signal_worker_terminate(worker_terminate);
            return;
        } else {
            spdlog::warn(rv.message);
        }
    }

    spdlog::debug("[separate_decode_data] Finished separating the decoded variant output.");
}

namespace {

secondary::IntervalTreesInt64Map make_variant_trees(
        const std::vector<secondary::Variant>& variants) {
    // Create intervals.
    std::unordered_map<int32_t, std::vector<secondary::IntervalInt64>> intervals_by_seq;
    for (const secondary::Variant& variant : variants) {
        const int64_t ref_end = variant_end(variant);
        // IntervalTree stops are inclusive; the value stores the exclusive REF end.
        intervals_by_seq[variant.seq_id].emplace_back(variant.pos, ref_end - 1, ref_end);
    }

    // Create trees from the intervals.
    secondary::IntervalTreesInt64Map trees;
    trees.reserve(std::size(intervals_by_seq));
    for (auto& [seq_id, intervals] : intervals_by_seq) {
        trees.emplace(seq_id, secondary::IntervalTreeInt64(std::move(intervals)));
    }

    return trees;
}

std::vector<secondary::IntervalInt64> make_processed_regions(
        const std::vector<secondary::VariantCallingSample>& vc_samples,
        const bool merge_intervals) {
    std::vector<secondary::IntervalInt64> intervals;
    for (const auto& vc_sample : vc_samples) {
        const int64_t start = vc_sample.start();  // Zero-based.
        const int64_t end = vc_sample.end() - 1;  // Exclusive.
        if (start > end) {
            continue;
        }
        intervals.emplace_back(start, end, vc_sample.seq_id);
    }

    std::sort(std::begin(intervals), std::end(intervals),
              [](const secondary::IntervalInt64& a, const secondary::IntervalInt64& b) {
                  return std::tie(a.value, a.start, a.stop) < std::tie(b.value, b.start, b.stop);
              });

    if (merge_intervals && !std::empty(intervals)) {
        std::vector<secondary::IntervalInt64> merged_intervals;
        merged_intervals.reserve(std::ssize(intervals));
        merged_intervals.emplace_back(intervals.front());
        for (const auto& iv : intervals) {
            // The +1 is because .stop is inclusive, and we want to merge neighboring regions.
            if (iv.start > (merged_intervals.back().stop + 1)) {
                merged_intervals.emplace_back(iv);
            }
            merged_intervals.back().stop = std::max(merged_intervals.back().stop, iv.stop);
        }
        std::swap(intervals, merged_intervals);
    } else {
        const auto new_end = std::unique(
                std::begin(intervals), std::end(intervals),
                [](const secondary::IntervalInt64& a, const secondary::IntervalInt64& b) {
                    return std::tie(a.value, a.start, a.stop) == std::tie(b.value, b.start, b.stop);
                });
        intervals.erase(new_end, intervals.end());
    }

    return intervals;
}

std::vector<secondary::IntervalInt64> construct_trimmed_merge_regions(
        const std::vector<secondary::VariantCallingSample>& vc_samples,
        const std::vector<secondary::Variant>& simple_variants,
        const int32_t flank_trim) {
    // Merge the processed regions. Only edges of these regions will be trimmed.
    // Non-const to avoid Clang-tidy error: constness of 'merged_processed_regions' prevents automatic move.
    std::vector<secondary::IntervalInt64> merged_processed_regions =
            make_processed_regions(vc_samples, true);

    // Nothing to do here.
    const int64_t trim = std::max<int32_t>(0, flank_trim);
    if (trim == 0) {
        return merged_processed_regions;
    }

    // Create variant trees to find any left-flank overlaps.
    const secondary::IntervalTreesInt64Map simple_variant_trees =
            make_variant_trees(simple_variants);

    const int32_t seq_id = vc_samples.front().seq_id;

    std::vector<secondary::IntervalInt64> merge_regions;

    for (const auto& iv : merged_processed_regions) {
        // Bluntly trim the coords left/right.
        int64_t new_start = iv.start + trim;
        int64_t new_end = iv.stop - trim;

        if (new_start > new_end) {
            continue;
        }

        // Find any simple variants on the left side which may overlap the trimmed sample coords.
        // Move the decode_start to the first position which doesn't overlap a simple variant.
        if (!std::empty(simple_variants)) {
            // Find the tree for this seq_id.
            const auto tree_iter = simple_variant_trees.find(seq_id);

            // Traverse all overlapping variants to move the new_start.
            while ((tree_iter != std::end(simple_variant_trees)) && (new_start < new_end)) {
                int64_t next_new_start = new_start;
                tree_iter->second.visit_overlapping(
                        new_start, new_start,
                        [&new_start,
                         &next_new_start](const secondary::IntervalInt64& variant_span) {
                            if (variant_span.start < new_start) {
                                // The +1 is because stop is inclusive.
                                next_new_start = std::max(next_new_start, variant_span.stop + 1);
                            }
                        });
                if (next_new_start <= new_start) {
                    break;
                }
                new_start = next_new_start;
            }

            // Traverse all overlapping variants to move the new_end in case the end is partially covering
            // a confident simple variant on the right flank.
            while ((tree_iter != std::end(simple_variant_trees)) && (new_start < new_end)) {
                int64_t next_new_end = new_end;
                tree_iter->second.visit_overlapping(
                        new_end, new_end,
                        [&new_end, &next_new_end](const secondary::IntervalInt64& variant_span) {
                            if (variant_span.start <= new_end) {
                                next_new_end = std::min(next_new_end, variant_span.start - 1);
                            }
                        });
                if (next_new_end >= new_end) {
                    break;
                }
                new_end = next_new_end;
            }

            if (new_start > new_end) {
                continue;
            }
        }

        merge_regions.emplace_back(secondary::IntervalInt64(new_start, new_end, iv.value));
    }

    return merge_regions;
}

std::vector<secondary::VariantCallingSample> trim_samples_to_merge_regions(
        const std::vector<secondary::VariantCallingSample>& vc_samples,
        const std::vector<secondary::IntervalInt64>& merge_regions) {
    if (std::empty(vc_samples) || std::empty(merge_regions)) {
        return {};
    }

    // Create trees from the intervals.
    std::vector<secondary::IntervalInt64> intervals = merge_regions;
    const secondary::IntervalTreeInt64 tree(std::move(intervals));

    std::vector<secondary::VariantCallingSample> ret;

    for (const auto& sample : vc_samples) {
        if (sample.start() >= sample.end()) {
            continue;
        }

        // Find overlapping merge regions.
        const std::vector<secondary::IntervalInt64> hits =
                tree.findOverlapping(sample.start(), sample.end() - 1);

        // This sample is not in any merge region, skip it.
        if (std::empty(hits)) {
            continue;
        }

        // Find the maximum start and minimum end coordinate of all overlapping merge regions.
        const auto max_it_start =
                std::max_element(std::begin(hits), std::end(hits),
                                 [](const auto& a, const auto& b) { return a.start < b.start; });
        const auto min_it_end =
                std::min_element(std::begin(hits), std::end(hits),
                                 [](const auto& a, const auto& b) { return a.stop < b.stop; });

        // Find the reference clip coordinates.
        const int64_t decode_start = std::max(max_it_start->start, sample.start());
        const int64_t decode_end = std::min(min_it_end->stop + 1, sample.end());

        if (decode_start >= decode_end) {
            continue;
        }

        // Clip.
        ret.emplace_back(slice_vc_sample_in_ref_coords(sample, decode_start, decode_end));
    }

    return ret;
}

std::pair<std::vector<secondary::Variant>, std::vector<secondary::IntervalInt64>>
call_variants_single_chrom(const std::string& draft,
                           const std::vector<secondary::VariantCallingSample>& vc_input_data,
                           const secondary::DecoderBase& decoder,
                           const std::vector<secondary::Variant>& simple_variants,
                           const int32_t flank_trim,
                           const float pass_min_qual,
                           const bool ambig_ref,
                           const bool gvcf) {
    if (std::empty(vc_input_data)) {
        return {};
    }

    std::vector<int32_t> ordered_ids;
    ordered_ids.reserve(std::size(vc_input_data));

    // Skip filtered samples.
    for (int32_t i = 0; i < std::ssize(vc_input_data); ++i) {
        if (vc_input_data[i].seq_id < 0) {
            continue;
        }
        ordered_ids.emplace_back(i);
    }

    // Sort by the full starting coordinate so async arrival order does not leak into trimming.
    std::stable_sort(std::begin(ordered_ids), std::end(ordered_ids),
                     [&vc_input_data](const int32_t lhs_id, const int32_t rhs_id) {
                         return secondary::variant_calling_sample_less(vc_input_data[lhs_id],
                                                                       vc_input_data[rhs_id]);
                     });

    if (std::empty(ordered_ids)) {
        return {};
    }

    // Debug print.
    if constexpr (DEBUG_VC_SAMPLES == true) {
        const int32_t seq_id = vc_input_data.front().seq_id;
        spdlog::trace("[call_variants_single_chrom seq_id = {}] vc_input_data: size = {}", seq_id,
                      std::size(vc_input_data));
        for (int64_t i = 0; i < std::ssize(vc_input_data); ++i) {
            spdlog::trace("[call_variants_single_chrom seq_id = {}]    [i = {}] {}", seq_id, i,
                          secondary::to_string(vc_input_data[i], false));
        }
    }

    // Trim the overlapping portions between samples.
    const auto trimmed_vc_samples = secondary::trim_vc_samples(vc_input_data, ordered_ids);

    // Debug print.
    if constexpr (DEBUG_VC_SAMPLES == true) {
        const int32_t seq_id = vc_input_data.front().seq_id;
        spdlog::trace("[call_variants_single_chrom seq_id = {}] trimmed_vc_samples: size = {}",
                      seq_id, std::size(trimmed_vc_samples));
        for (int64_t i = 0; i < std::ssize(trimmed_vc_samples); ++i) {
            spdlog::trace("[call_variants_single_chrom seq_id = {}]    [i = {}] {}", seq_id, i,
                          secondary::to_string(trimmed_vc_samples[i], false));
        }
    }

    const std::vector<secondary::IntervalInt64> merge_regions =
            construct_trimmed_merge_regions(trimmed_vc_samples, simple_variants, flank_trim);

    // Trim overlaps with simple variants.
    const auto callable_vc_samples =
            trim_samples_to_merge_regions(trimmed_vc_samples, merge_regions);

    // Debug print.
    if constexpr (DEBUG_VC_SAMPLES == true) {
        const int32_t seq_id = vc_input_data.front().seq_id;
        spdlog::trace("[call_variants_single_chrom seq_id = {}] callable_vc_samples: size = {}",
                      seq_id, std::size(callable_vc_samples));
        for (int64_t i = 0; i < std::ssize(callable_vc_samples); ++i) {
            spdlog::trace("[call_variants_single_chrom seq_id = {}]    [i = {}] {}", seq_id, i,
                          secondary::to_string(callable_vc_samples[i], false));
        }
    }

    // Break and merge samples on non-variant positions.
    const auto joined_samples = join_samples(callable_vc_samples, draft, decoder);
    std::vector<secondary::Variant> results;

    for (const auto& vc_sample : joined_samples) {
        std::vector<secondary::Variant> variants = secondary::general_decode_variants(
                decoder, vc_sample.seq_id, vc_sample.positions_major, vc_sample.positions_minor,
                vc_sample.logits, draft, pass_min_qual, ambig_ref, gvcf, true, true, false,
                secondary::DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);

        results.insert(std::end(results), std::make_move_iterator(std::begin(variants)),
                       std::make_move_iterator(std::end(variants)));
    }

    // Sort the variants.
    std::stable_sort(std::begin(results), std::end(results));

    return {results, merge_regions};
}

std::vector<secondary::Interval64> merge_intervals(
        const std::vector<secondary::RegionInt>& intervals,
        const int64_t seq_len) {
    std::vector<secondary::Interval64> valid_intervals;
    valid_intervals.reserve(std::size(intervals));

    for (const auto& interval : intervals) {
        if (!secondary::is_valid(interval)) {
            continue;
        }
        const secondary::RegionInt region = secondary::normalize_region(interval, seq_len);
        if (!secondary::is_valid(region)) {
            continue;
        }
        valid_intervals.emplace_back(secondary::Interval64{region.start, region.end});
    }

    std::sort(std::begin(valid_intervals), std::end(valid_intervals));

    std::vector<secondary::Interval64> merged;
    merged.reserve(std::size(valid_intervals));
    for (auto&& interval : valid_intervals) {
        if (std::empty(merged) || (interval.start > merged.back().end)) {
            merged.emplace_back(std::move(interval));
            continue;
        }
        merged.back().end = std::max(merged.back().end, interval.end);
    }

    return merged;
}

secondary::Variant make_gvcf_reference_record(const int32_t seq_id,
                                              const int64_t pos,
                                              const char ref_base,
                                              const int64_t end,
                                              const int32_t ploidy) {
    const float qual = utils::is_canonical_base(ref_base) ? secondary::VCF_MAX_GQ_CAP : 0.0f;
    const secondary::Variant ref_var{
            .seq_id = seq_id,
            .pos = pos,
            .ref = std::string(1, ref_base),
            .alts = {"."},
            .filter = ".",
            .info = {},
            .qual = qual,
            .genotype = {},
            .rstart = pos,
            .rend = end,
    };

    secondary::Variant ret =
            secondary::normalize_genotype(ref_var, ploidy, secondary::VCF_MAX_GQ_CAP);

    secondary::set_gvcf_reference_block_end(ret, end);

    return ret;
}

std::vector<secondary::Variant> compact_gvcf_reference_records(
        const std::vector<secondary::Variant>& variants,
        const std::span<const std::pair<int64_t, float>> gq_margins) {
    const auto is_single_base_gvcf_reference_record = [](const secondary::Variant& var) {
        return (var.filter == ".") && (std::size(var.alts) == 1) && (var.alts.front() == ".") &&
               std::empty(var.info) && (std::size(var.ref) == 1);
    };

    const auto is_same_genotype = [](const secondary::Variant& lhs, const secondary::Variant& rhs) {
        const std::string* lhs_gt = nullptr;
        const std::string* rhs_gt = nullptr;

        for (const auto& val : lhs.genotype) {
            if (val.first == "GT") {
                lhs_gt = &val.second;
                break;
            }
        }
        for (const auto& val : rhs.genotype) {
            if (val.first == "GT") {
                rhs_gt = &val.second;
                break;
            }
        }

        return lhs_gt && rhs_gt && (*lhs_gt == *rhs_gt);
    };

    const auto set_gq = [](secondary::Variant& var, const float gq) {
        var.qual = gq;

        const std::string gq_str = std::to_string(static_cast<int32_t>(std::round(gq)));
        for (auto val = std::rbegin(var.genotype); val != std::rend(var.genotype); ++val) {
            if (val->first == "GQ") {
                if (val->second != ".") {
                    val->second = gq_str;
                }
                break;
            }
        }
    };

    std::vector<secondary::Variant> compacted;
    compacted.reserve(std::size(variants));

    std::size_t i = 0;
    while (i < std::size(variants)) {
        if (!is_single_base_gvcf_reference_record(variants[i])) {
            compacted.emplace_back(variants[i]);
            ++i;
            continue;
        }

        const float first_gq = variants[i].qual;
        const float gq_margin =
                secondary::get_gvcf_reference_record_gq_margin(first_gq, gq_margins);
        float min_gq = first_gq;

        std::size_t j = i + 1;
        while ((j < std::size(variants)) && is_single_base_gvcf_reference_record(variants[j]) &&
               (variants[j].seq_id == variants[i].seq_id) &&
               (variants[j].pos == (variants[j - 1].pos + 1))) {
            const float current_gq = variants[j].qual;
            if ((std::fabs(current_gq - first_gq) > gq_margin) ||
                !is_same_genotype(variants[j], variants[i])) {
                break;
            }
            min_gq = std::min(min_gq, current_gq);
            ++j;
        }

        secondary::Variant block = variants[i];
        secondary::set_gvcf_reference_block_end(block, variants[j - 1].pos + 1);
        set_gq(block, min_gq);
        compacted.emplace_back(std::move(block));
        i = j;
    }

    return compacted;
}

void add_gvcf_reference_records_for_unprocessed_regions(
        std::vector<secondary::Variant>& variants,
        const int32_t seq_id,
        const std::string& draft,
        const std::vector<secondary::RegionInt>& selected_regions,
        const int32_t ploidy) {
    const int64_t seq_len = std::ssize(draft);
    const std::vector<secondary::Interval64> selected = merge_intervals(selected_regions, seq_len);
    const std::size_t num_existing_variants = std::size(variants);
    std::size_t variant_idx = 0;

    for (const secondary::Interval64& selected_interval : selected) {
        int64_t pos = selected_interval.start;
        while (pos < selected_interval.end) {
            while ((variant_idx < num_existing_variants) &&
                   (!secondary::is_valid(variants[variant_idx]) ||
                    (variants[variant_idx].seq_id < seq_id) ||
                    ((variants[variant_idx].seq_id == seq_id) &&
                     secondary::variant_ends_before_position(variants[variant_idx], seq_id,
                                                             pos)))) {
                ++variant_idx;
            }

            if ((variant_idx < num_existing_variants) &&
                secondary::variant_covers_position(variants[variant_idx], seq_id, pos)) {
                pos = std::min<int64_t>(selected_interval.end,
                                        secondary::variant_end(variants[variant_idx]));
                continue;
            }

            const int64_t block_start = pos;
            const bool is_callable_block =
                    utils::is_canonical_base(draft[static_cast<std::size_t>(block_start)]);
            pos = selected_interval.end;
            if ((variant_idx < num_existing_variants) && (variants[variant_idx].seq_id == seq_id)) {
                pos = std::min(pos, variants[variant_idx].pos);
            }
            for (int64_t candidate_pos = block_start + 1; candidate_pos < pos; ++candidate_pos) {
                if (utils::is_canonical_base(draft[static_cast<std::size_t>(candidate_pos)]) !=
                    is_callable_block) {
                    pos = candidate_pos;
                    break;
                }
            }

            variants.emplace_back(make_gvcf_reference_record(
                    seq_id, block_start, draft[static_cast<std::size_t>(block_start)], pos,
                    ploidy));
        }
    }
}

std::vector<secondary::Variant> merge_variants(
        const std::vector<secondary::Variant>& inference_variants,
        const std::vector<secondary::Variant>& simple_variants,
        const std::vector<secondary::IntervalInt64>& inference_regions_to_keep) {
    // Create the lookup tree for filtering.
    std::vector<secondary::IntervalInt64> intervals = inference_regions_to_keep;
    const secondary::IntervalTreeInt64 tree = secondary::IntervalTreeInt64(std::move(intervals));

    std::vector<secondary::Variant> new_variants;
    new_variants.reserve(std::size(inference_variants) + std::size(simple_variants));

    // Keep inference variants which are within the inference_regions_to_keep.
    for (const secondary::Variant& var : inference_variants) {
        const std::vector<interval_tree::Interval<int64_t, int64_t>> region_hits =
                tree.findOverlapping(var.pos, var.pos);
        if (std::empty(region_hits)) {
            continue;
        }
        new_variants.emplace_back(var);
    }

    // Keep Kadayashi variants which are not within the inference_regions_to_keep.
    // IMPORTANT: Any Kadayashi variant which overlaps inference_regions_to_keep will be removed.
    for (const secondary::Variant& var : simple_variants) {
        const int64_t end = std::max(var.pos, secondary::variant_end(var) - 1);
        const std::vector<interval_tree::Interval<int64_t, int64_t>> region_hits =
                tree.findOverlapping(var.pos, end);
        // IMPORTANT different from the above block - negative test.
        if (!std::empty(region_hits)) {
            continue;
        }
        new_variants.emplace_back(var);
    }

    std::stable_sort(std::begin(new_variants), std::end(new_variants));

    return new_variants;
}

}  // namespace

std::vector<secondary::Variant> filter_hemizygous_variants(
        const std::vector<secondary::Variant>& variants,
        const std::vector<secondary::Region>& hemizygous_regions) {
    if (std::empty(hemizygous_regions)) {
        return variants;
    }

    std::vector<secondary::Variant> filtered_variants;
    filtered_variants.reserve(std::ssize(variants));
    std::vector<secondary::Region>::const_iterator iter_regions = hemizygous_regions.begin();
    for (const secondary::Variant& var : variants) {
        secondary::Variant new_var = var;
        const int64_t var_start = var.pos;
        const int64_t var_end = var.pos + var.ref.length();
        while ((iter_regions != hemizygous_regions.end()) && (iter_regions->end > 0) &&
               (iter_regions->end <= var_start)) {
            ++iter_regions;
        }
        if ((iter_regions != hemizygous_regions.end()) && (iter_regions->start < var_end)) {
            // variant overlaps the region end
            if ((var_start < iter_regions->start) ||
                ((var_end > iter_regions->end) && (iter_regions->end > 0))) {
                spdlog::debug(
                        "[filter_hemizygous_variants] Variant {} {} overlaps hemizygous region "
                        "end, leaving it unchanged.",
                        iter_regions->name, var.pos);
            } else {
                new_var = secondary::collapse_to_haploid(var, true);
            }
        }
        if (is_valid(new_var)) {
            filtered_variants.emplace_back(std::move(new_var));
        }
    }
    return filtered_variants;
}

void worker_variant_calling_reduce(
        utils::AsyncQueue<secondary::VariantCallingSample>& input_queue,
        utils::AsyncQueue<int64_t>& out_queue,  // ID of the chromosome that is ready for writing.
        std::vector<ChromosomeReduceData>& chrom_reduce_data,
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        secondary::Stats& stats,
        const std::vector<std::unique_ptr<hts_io::FastxRandomReader>>& fastx_readers,
        const bool continue_on_exception,
        const int32_t num_threads,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const secondary::DecoderBase& decoder,
        const std::vector<std::vector<secondary::Region>>& hemizygous_regions,
        const float pass_min_qual,
        const bool ambig_ref,
        const bool gvcf,
        const int32_t ploidy,
        const int32_t flank_trim,
        const secondary::VariantCandidateSource variant_candidate_source) {
    utils::ScopedProfileRange spr1("variant_calling_reduce", 2);

    using Bool = uint8_t;

    std::vector<std::vector<secondary::VariantCallingSample>> chrom_vc_samples(
            std::size(chrom_reduce_data));
    std::vector<Bool> chrom_reduced(std::size(chrom_reduce_data), false);
    std::vector<int64_t> chrom_progress(std::size(chrom_reduce_data), 0);

    const auto reduce_ready_chromosome = [&](const int32_t seq_id, const int32_t tid,
                                             std::unique_lock<std::mutex>& lock,
                                             const bool allow_incomplete) {
        if (!lock.owns_lock()) {
            throw std::runtime_error{
                    "Expected the chromosome reduction mutex to be held before processing."};
        }

        auto& reduce_data = chrom_reduce_data[seq_id];
        const bool all_expected_samples_received =
                (std::ssize(chrom_vc_samples[seq_id]) == reduce_data.num_samples);

        // Skip merging until all data is ready.
        if (chrom_reduced[seq_id] || !reduce_data.ready ||
            (!allow_incomplete && !all_expected_samples_received)) {
            return;
        }

        if (allow_incomplete && !all_expected_samples_received) {
            spdlog::warn(
                    "[variant_calling_reduce tid = {}] Finalizing sequence '{}' with partial "
                    "variant-calling input under continue-on-error. received = {}, expected = {}",
                    tid, reduce_data.seq_name, std::size(chrom_vc_samples[seq_id]),
                    reduce_data.num_samples);
        }

        const std::vector<secondary::VariantCallingSample>& vc_decode_data =
                chrom_vc_samples[seq_id];

        const std::string& seq_name = draft_lens[seq_id].first;
        std::optional<std::string> draft;
        if (!std::empty(vc_decode_data) || gvcf) {
            draft.emplace(fastx_readers[tid]->fetch_seq(seq_name));
        }

        std::vector<secondary::Variant> variants;
        std::vector<secondary::IntervalInt64> merge_regions;
        if (!std::empty(vc_decode_data)) {
            const bool merge_with_simple =
                    (variant_candidate_source == secondary::VariantCandidateSource::COMPUTE);
            const std::vector<secondary::Variant> empty_simple_variants;
            const auto& simple_variants =
                    merge_with_simple ? reduce_data.variants_simple : empty_simple_variants;
            std::tie(variants, merge_regions) = call_variants_single_chrom(
                    *draft, vc_decode_data, decoder, simple_variants,
                    merge_with_simple ? flank_trim : 0, pass_min_qual, ambig_ref, gvcf);
        }

        spdlog::debug(
                "[variant_calling_reduce tid = {}] Finished calling variants "
                "for sequence: {}, variants: {}, "
                "reduce_data.ready = {}, reduce_data.num_samples = {}",
                tid, seq_name, std::size(variants), reduce_data.ready, reduce_data.num_samples);

        reduce_data.processed_regions = make_processed_regions(vc_decode_data, false);

        // Store inference variants.
        reduce_data.variants_inference = std::move(variants);

        // Only merge variants if confident variants were computed internally.
        if (variant_candidate_source == secondary::VariantCandidateSource::COMPUTE) {
            reduce_data.variants_merged = merge_variants(
                    reduce_data.variants_inference, reduce_data.variants_simple, merge_regions);
        } else {
            reduce_data.variants_merged = reduce_data.variants_inference;
        }
        if (gvcf) {
            std::stable_sort(std::begin(reduce_data.variants_merged),
                             std::end(reduce_data.variants_merged));
            add_gvcf_reference_records_for_unprocessed_regions(reduce_data.variants_merged, seq_id,
                                                               *draft, reduce_data.selected_regions,
                                                               ploidy);
            std::stable_sort(std::begin(reduce_data.variants_merged),
                             std::end(reduce_data.variants_merged));
            reduce_data.variants_merged = compact_gvcf_reference_records(
                    reduce_data.variants_merged,
                    secondary::DEFAULT_GVCF_REFERENCE_BLOCK_GQ_MARGINS);
        }

        if (!std::empty(hemizygous_regions)) {
            const std::vector<secondary::Region>& reduce_hemizygous = hemizygous_regions[seq_id];
            if (!std::empty(reduce_hemizygous)) {
                reduce_data.variants_merged =
                        filter_hemizygous_variants(reduce_data.variants_merged, reduce_hemizygous);
            }
        }

        // Release the data after processing.
        chrom_vc_samples[seq_id].clear();
        chrom_reduced[seq_id] = true;

        // Trigger writing.
        out_queue.try_push(seq_id);

        // Top up any remaining stage-2 budget, including chromosomes which produced no
        // inference samples at all.
        const int64_t remaining_progress =
                std::max<int64_t>(0, reduce_data.progress_target - chrom_progress[seq_id]);
        stats.add("processed", static_cast<double>(remaining_progress));
        chrom_progress[seq_id] += remaining_progress;
    };

    const auto worker = [&](const int32_t tid, secondary::WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("variant_calling_reduce-worker", 3);
        at::InferenceMode infer_guard;

        while (!worker_terminate) {
            utils::ScopedProfileRange spr3("variant_calling_reduce-worker-while", 4);

            secondary::VariantCallingSample item;
            const auto pop_status = input_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            try {
                const int32_t seq_id = item.seq_id;

                // Sanity check.
                if (seq_id >= std::ssize(chrom_reduce_data)) {
                    throw std::runtime_error{
                            "Variant calling sample's seq_id is larger than the expected number of "
                            "input sequences. seq_id = " +
                            std::to_string(seq_id) + ", chrom_reduce_data.size = " +
                            std::to_string(std::size(chrom_reduce_data))};
                }

                auto& reduce_data = chrom_reduce_data[seq_id];

                // Add the data to an appropriate spot and process it.
                {
                    std::unique_lock<std::mutex> lock(reduce_data.mtx);

                    // Approximate stage-2 progress from incoming inferred samples, but never let
                    // the per-sequence total exceed the selected input span.
                    const int64_t sample_progress = std::max<int64_t>(0, item.end() - item.start());
                    const int64_t remaining_progress = std::max<int64_t>(
                            0, reduce_data.progress_target - chrom_progress[seq_id]);
                    const int64_t progress_increment =
                            std::min(sample_progress, remaining_progress);
                    stats.add("processed", static_cast<double>(progress_increment));
                    chrom_progress[seq_id] += progress_increment;

                    // Move the item to the corresponding chromosome.
                    chrom_vc_samples[seq_id].emplace_back(std::move(item));
                    reduce_ready_chromosome(seq_id, tid, lock, false);
                }

            } catch (const std::exception& e) {
                if (!continue_on_exception) {
                    ret_val = {.exception_thrown = true,
                               .message = std::string("Caught exception while reducing the variant "
                                                      "calling results: '" +
                                                      std::string(e.what()) + "'")};
                    signal_worker_terminate(worker_terminate);
                    return;
                }

                spdlog::warn(
                        "Caught an exception while reducing the variant calling results. Skipping "
                        "this "
                        "batch. Exception: {}",
                        e.what());
            }
        }
    };

    const int64_t actual_num_threads = std::min<int64_t>(std::ssize(fastx_readers), num_threads);

    cxxpool::thread_pool pool{static_cast<size_t>(actual_num_threads)};

    std::vector<secondary::WorkerReturnStatus> worker_return_vals(num_threads);

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    // A chromosome can become ready after its final VariantCallingSample has already been consumed,
    // for example when the last BAM window for that chromosome produced zero samples.
    // Alternatively, when the input chromosome has zero samples, we need to "merge" the
    // simple variants with (empty) inference variants, otherwise some data might go missing.
    // This finalizes any such chromosomes once the input stream is drained.
    for (int32_t seq_id = 0; seq_id < std::ssize(chrom_reduce_data); ++seq_id) {
        try {
            auto& reduce_data = chrom_reduce_data[seq_id];
            std::unique_lock<std::mutex> lock(reduce_data.mtx);
            reduce_ready_chromosome(seq_id, 0, lock, continue_on_exception);
        } catch (const std::exception& e) {
            if (!continue_on_exception) {
                ret_status = {.exception_thrown = true,
                              .message = std::string(
                                      "Caught exception while finalizing the variant calling "
                                      "results: '" +
                                      std::string(e.what()) + "'")};
                signal_worker_terminate(worker_terminate);
                return;
            }

            spdlog::warn(
                    "Caught an exception while finalizing the variant calling results. Skipping "
                    "this chromosome. Exception: {}",
                    e.what());
        }
    }

    if (!worker_terminate.load(std::memory_order_acquire)) {
        out_queue.terminate(utils::AsyncQueueTerminateFast::No);
    }

    for (const secondary::WorkerReturnStatus& rv : worker_return_vals) {
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            signal_worker_terminate(worker_terminate);
            return;
        } else {
            spdlog::warn(rv.message);
        }
    }

    spdlog::debug("[variant_calling_reduce] Finished reducing the decoded variant output.");
}

void worker_variant_writer(
        utils::AsyncQueue<int64_t>& input_queue,  // ID of the chromosome that is ready for writing.
        std::vector<ChromosomeReduceData>&
                chrom_reduce_data,  // Non-const so that data can be freed.
        std::atomic<bool>& worker_terminate,
        secondary::WorkerReturnStatus& ret_status,
        secondary::VCFWriter& vcf_writer,
        std::ofstream& ofs_regions,
        std::optional<secondary::VCFWriter>& vcf_writer_simple,
        std::optional<secondary::VCFWriter>& vcf_writer_inference,
        const bool continue_on_exception) {
    utils::ScopedProfileRange spr1("variant_writer", 2);

    const auto worker = [&](const int32_t tid, secondary::WorkerReturnStatus& ret_val) {
        utils::ScopedProfileRange spr2("variant_writer-worker", 3);
        at::InferenceMode infer_guard;

        while (!worker_terminate) {
            utils::ScopedProfileRange spr3("variant_writer-worker-while", 4);

            int64_t item{0};
            const auto pop_status = input_queue.try_pop(item);

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            try {
                const int32_t seq_id = item;

                // Sanity check.
                if (seq_id >= std::ssize(chrom_reduce_data)) {
                    throw std::runtime_error{
                            "Chromosome seq_id is larger than the expected number of "
                            "input sequences when attempting to write the results. seq_id = " +
                            std::to_string(seq_id) + ", chrom_reduce_data.size = " +
                            std::to_string(std::size(chrom_reduce_data))};
                }

                // Non-const so we can free the results for this chromosome.
                auto& results = chrom_reduce_data[seq_id];

                spdlog::debug(
                        "[variant_writer tid = {}] Writing variants for sequence: {}, seq_id = "
                        "{}",
                        tid, results.seq_name, seq_id);

                // Write the results and free the data.
                {
                    std::unique_lock<std::mutex> lock(results.mtx);

                    // Write the VCF file.
                    for (const secondary::Variant& variant : results.variants_merged) {
                        vcf_writer.write_variant(variant);
                    }

                    // Write the processed_regions.bed.
                    if (ofs_regions.is_open()) {
                        for (const secondary::IntervalInt64 iv : results.processed_regions) {
                            ofs_regions << results.seq_name << '\t' << iv.start << '\t'
                                        << (iv.stop + 1) << '\n';
                        }
                    }

                    // Write Kadayashi VCF.
                    if (vcf_writer_simple) {
                        for (const secondary::Variant& variant : results.variants_simple) {
                            vcf_writer_simple->write_variant(variant);
                        }
                    }

                    // Write the inference VCF file.
                    if (vcf_writer_inference) {
                        for (const secondary::Variant& variant : results.variants_inference) {
                            vcf_writer_inference->write_variant(variant);
                        }
                    }

                    // Release per-chromosome output buffers once they have been handed off to the
                    // output streams so later chromosomes can reuse the memory.
                    std::vector<secondary::Variant>{}.swap(results.variants_merged);
                    std::vector<secondary::Variant>{}.swap(results.variants_simple);
                    std::vector<secondary::Variant>{}.swap(results.variants_inference);
                    std::vector<secondary::RegionInt>{}.swap(results.selected_regions);
                    std::vector<secondary::IntervalInt64>{}.swap(results.processed_regions);
                }

            } catch (const std::exception& e) {
                if (!continue_on_exception) {
                    ret_val = {.exception_thrown = true,
                               .message = std::string("Caught exception while writing variants: '" +
                                                      std::string(e.what()) + "'")};
                    signal_worker_terminate(worker_terminate);
                    return;
                }

                spdlog::warn(
                        "Caught an exception while writing variants. Skipping this "
                        "item. Exception: {}",
                        e.what());
            }
        }
    };

    constexpr size_t NUM_THREADS = 1;

    cxxpool::thread_pool pool{static_cast<size_t>(NUM_THREADS)};

    std::vector<secondary::WorkerReturnStatus> worker_return_vals(NUM_THREADS);

    std::vector<std::future<void>> futures;
    futures.reserve(NUM_THREADS);

    for (int32_t tid = 0; tid < static_cast<int32_t>(NUM_THREADS); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(worker_return_vals[tid])));
    }

    for (auto& f : futures) {
        f.get();
    }

    for (const secondary::WorkerReturnStatus& rv : worker_return_vals) {
        if (!rv.exception_thrown) {
            continue;
        }
        if (!continue_on_exception) {
            // Cannot throw because this is a worker function intended to run on a separate thread.
            // Instead, communicate the error and return.
            ret_status = rv;
            signal_worker_terminate(worker_terminate);
            return;
        } else {
            spdlog::warn(rv.message);
        }
    }

    spdlog::debug("[variant_writer] Finished writing the variant output.");
}

}  // namespace dorado::smallvar
