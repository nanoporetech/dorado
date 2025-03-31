#include "CorrectionInferenceNode.h"

#include "correct/conversions.h"
#include "correct/decode.h"
#include "correct/features.h"
#include "correct/infer.h"
#include "correct/windows.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/bam_utils.h"
#include "utils/paf_utils.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/thread_naming.h"
#include "utils/types.h"

#include <stdexcept>
#include <unordered_set>
#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif
#include "hts_io/FastxRandomReader.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <ATen/Tensor.h>
#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <minimap.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#if DORADO_CUDA_BUILD
constexpr bool USING_DORADO_CUDA_BUILD = true;
#else
constexpr bool USING_DORADO_CUDA_BUILD = false;
#endif

using namespace dorado::correction;

// #define DEBUG_CORRECT_PRINT_WINDOW_SIZES_TO_FILE
// #define DEBUG_CORRECT_PRINT_WINDOW_INFO_TO_FILE

namespace {

dorado::BamPtr create_bam_record(const std::string& read_id, const std::string& seq) {
    bam1_t* rec = bam_init1();
    bam_set1(rec, read_id.length(), read_id.c_str(), 4 /*flag*/, -1 /*tid*/, -1 /*pos*/, 0 /*mapq*/,
             0 /*n_cigar*/, nullptr /*cigar*/, -1 /*mtid*/, -1 /*mpos*/, 0 /*isize*/, seq.size(),
             seq.data(), nullptr, 0);
    return dorado::BamPtr(rec);
}

bool populate_alignments(dorado::CorrectionAlignments& alignments,
                         dorado::hts_io::FastxRandomReader* reader,
                         const std::unordered_set<int>& useful_overlap_idxs) {
    const auto& tname = alignments.read_name;

    alignments.read_seq = reader->fetch_seq(tname);
    alignments.read_qual = reader->fetch_qual(tname);
    int tlen = (int)alignments.read_seq.length();

    // Might be worthwhile generating dense vectors with some index mapping to save memory
    // as using filtering of useful overlaps makes these vectors sparse.
    auto num_qnames = alignments.qnames.size();
    alignments.seqs.resize(num_qnames);
    alignments.quals.resize(num_qnames);
    alignments.cigars.resize(num_qnames);

    for (const size_t i : useful_overlap_idxs) {
        const std::string& qname = alignments.qnames[i];
        alignments.seqs[i] = reader->fetch_seq(qname);
        if ((int)alignments.seqs[i].length() != alignments.overlaps[i].qlen) {
            spdlog::error("qlen from before {} and qlen from after {} don't match for {}",
                          alignments.overlaps[i].qlen, alignments.seqs[i].length(), qname);
            return false;
        }
        alignments.quals[i] = reader->fetch_qual(qname);
        if (alignments.overlaps[i].tlen != tlen) {
            spdlog::error("tlen from before {} and tlen from after {} don't match for {}",
                          alignments.overlaps[i].tlen, tlen, tname);
            return false;
        }
    }

    return alignments.check_consistent_overlaps();
}

std::vector<std::string> concatenate_corrected_windows(const std::vector<std::string>& cons) {
    std::vector<std::string> corrected_seqs;

    std::string corrected_seq = "";

    for (const auto& s : cons) {
        if (s.empty()) {
            if (!corrected_seq.empty()) {
                corrected_seqs.push_back(std::move(corrected_seq));
                corrected_seq = "";
            }
        } else {
            corrected_seq += s;
        }
    }
    if (!corrected_seq.empty()) {
        corrected_seqs.push_back(std::move(corrected_seq));
    }
    return corrected_seqs;
}

}  // namespace

namespace dorado {

void CorrectionInferenceNode::concat_features_and_send(const std::vector<std::string>& to_decode,
                                                       const std::string& read_name) {
    LOG_TRACE("decoding window for {}", read_name);
    auto corrected_seqs = concatenate_corrected_windows(to_decode);
    if (corrected_seqs.size() == 1) {
        BamMessage rec{create_bam_record(read_name, corrected_seqs[0]), nullptr};
        send_message_to_sink(std::move(rec));
    } else {
        for (size_t s = 0; s < corrected_seqs.size(); s++) {
            const std::string new_name = read_name + ":" + std::to_string(s);
            BamMessage rec{create_bam_record(new_name, corrected_seqs[s]), nullptr};
            send_message_to_sink(std::move(rec));
        }
    }
}

void CorrectionInferenceNode::decode_fn() {
    utils::set_thread_name("corr_decode");
    spdlog::debug("Starting decode thread!");

    WindowFeatures item;
    while (m_inferred_features_queue.try_pop(item) != utils::AsyncQueueStatus::Terminate) {
        utils::ScopedProfileRange spr("decode_loop", 1);
        auto read_name = item.read_name;
        std::vector<std::string> to_decode;
        auto pos = item.window_idx;
        auto corrected_seq = decode_window(item);
        {
            std::lock_guard<std::mutex> lock(m_features_mutex);
            auto find_iter = m_features_by_id.find(read_name);
            if (find_iter == m_features_by_id.end()) {
                spdlog::error("Decoded feature list not found for {}.", read_name);
                continue;
            }
            auto& output_features = find_iter->second;
            output_features[pos] = std::move(corrected_seq);
            auto& pending = m_pending_features_by_id.find(read_name)->second;
            pending--;
            if (pending == 0) {
                // Got all features!
                to_decode = std::move(output_features);
                m_features_by_id.erase(read_name);
                m_pending_features_by_id.erase(read_name);
            }
        }

        if (!to_decode.empty()) {
            concat_features_and_send(to_decode, read_name);
        }
    }
}

void CorrectionInferenceNode::infer_fn(const std::string& device_str, int mtx_idx, int batch_size) {
    utils::set_thread_name("corr_infer");
    spdlog::debug("Starting process thread for {}!", device_str);
    m_num_active_infer_threads++;

    torch::Device device = torch::Device(device_str);

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (device.is_cuda()) {
        c10::cuda::CUDAGuard device_guard(device);
        stream = c10::cuda::getStreamFromPool(false, device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    at::InferenceMode infer_guard;

    auto model_path = (m_model_config.model_dir / m_model_config.weights_file).string();
    torch::jit::script::Module module;
    try {
        spdlog::debug("Loading model on {}...", device_str);
        module = torch::jit::load(model_path, device);
        spdlog::debug("Loaded model on {}!", device_str);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path +
                                 " with error: " + e.what());
    }
    module.eval();

    std::vector<at::Tensor> bases_batch;
    std::vector<at::Tensor> quals_batch;
    std::vector<int> lengths;
    std::vector<int64_t> sizes;
    std::vector<at::Tensor> indices_batch;
    std::vector<WindowFeatures> wfs;
    // If there are any windows > 5120, then reduce batch size by 1
    int remaining_batch_slots = batch_size;

    auto decode_preds = [](const at::Tensor& preds) {
        std::vector<char> bases;
        bases.reserve(preds.sizes()[0]);
        static std::array<char, 5> decoder = {'A', 'C', 'G', 'T', '*'};
        for (int i = 0; i < preds.sizes()[0]; i++) {
            auto base_idx = preds[i].item<int>();
            bases.push_back(decoder[base_idx]);
        }
        return bases;
    };

    auto batch_infer = [&, this]() {
        utils::ScopedProfileRange infer("infer", 1);
        // Run inference on batch
        auto length_tensor =
                at::from_blob(lengths.data(), {(int)lengths.size()},
                              at::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

        const bool legacy_windowing = this->m_legacy_windowing;

        // Collate bases.
        at::Tensor batched_bases;
        std::thread thread_bases = std::thread([&bases_batch, &batched_bases, legacy_windowing]() {
            const bool use_pinned_memory = USING_DORADO_CUDA_BUILD && !legacy_windowing;
            batched_bases = correction::collate<int32_t>(bases_batch, static_cast<int32_t>(11),
                                                         torch::kInt32, use_pinned_memory);
            bases_batch.clear();
        });

        // Collate quals.
        at::Tensor batched_quals;
        std::thread thread_quals = std::thread([&quals_batch, &batched_quals, legacy_windowing]() {
            const bool use_pinned_memory = USING_DORADO_CUDA_BUILD && !legacy_windowing;
            batched_quals = correction::collate<float>(quals_batch, 0.0f, torch::kFloat32,
                                                       use_pinned_memory);
            quals_batch.clear();
        });

        thread_bases.join();
        thread_quals.join();

        std::vector<torch::jit::IValue> inputs;
        {
            const bool non_blocking = !m_legacy_windowing;
            utils::ScopedProfileRange move_to_device("move_to_device", 1);
            inputs.push_back(batched_bases.to(device, non_blocking));
            inputs.push_back(batched_quals.to(device, non_blocking));
            inputs.push_back(length_tensor.to(device, non_blocking));
            std::for_each(indices_batch.begin(), indices_batch.end(),
                          [device, non_blocking](at::Tensor& t) { t.to(device, non_blocking); });
            inputs.push_back(indices_batch);
        }

        std::unique_lock<std::mutex> lock(m_gpu_mutexes[mtx_idx]);

        c10::IValue output;
        try {
            output = module.forward(inputs);
        } catch (std::runtime_error& e) {
#if DORADO_CUDA_BUILD
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.what());
            c10::cuda::CUDACachingAllocator::emptyCache();
            output = module.forward(inputs);
#else
            throw e;
#endif
        }
        lock.unlock();
        if (!output.isTuple()) {
            throw std::runtime_error("Expected inference result to be tuple.");
        }
        auto base_logits = output.toTuple()->elements()[1].toTensor();
        auto preds = base_logits.argmax(1, false).to(torch::kCPU);
        auto split_preds = preds.split_with_sizes(sizes);
        for (size_t w = 0; w < split_preds.size(); w++) {
            auto decoded_output = decode_preds(split_preds[w]);
            wfs[w].inferred_bases = decoded_output;
        }

        for (auto& wf : wfs) {
            m_inferred_features_queue.try_push(std::move(wf));
        }

        bases_batch.clear();
        quals_batch.clear();
        lengths.clear();
        sizes.clear();
        wfs.clear();
        indices_batch.clear();
        remaining_batch_slots = batch_size;
    };

    WindowFeatures item;
    auto last_chunk_reserve_time = std::chrono::system_clock::now();
    while (true) {
        const auto pop_status = m_features_queue.try_pop_until(
                item, last_chunk_reserve_time + std::chrono::milliseconds(10000));

        if (pop_status == utils::AsyncQueueStatus::Terminate) {
            break;
        }

        if (pop_status == utils::AsyncQueueStatus::Timeout) {
            // Ended with a timeout, so run inference if there are samples.
            if (bases_batch.size() > 0) {
                batch_infer();
            }
            last_chunk_reserve_time = std::chrono::system_clock::now();
            continue;
        }

        utils::ScopedProfileRange spr("collect_features", 1);
        int required_batch_slots = ((int)item.bases.sizes()[1] / 5120) + 1;
        if (required_batch_slots > remaining_batch_slots) {
            batch_infer();
        }
        wfs.push_back(std::move(item));
        auto& wf = wfs.back();

        auto b = wf.bases.transpose(0, 1);
        auto q = wf.quals.transpose(0, 1);

        bases_batch.push_back(b);
        quals_batch.push_back(q);
        lengths.push_back(wf.length);
        sizes.push_back(wf.length);
        indices_batch.push_back(wf.indices);
        remaining_batch_slots -= required_batch_slots;
        last_chunk_reserve_time = std::chrono::system_clock::now();
    }

    if (bases_batch.size() > 0) {
        batch_infer();
    }

    auto remaining_threads = --m_num_active_infer_threads;
    if (remaining_threads == 0) {
        m_inferred_features_queue.terminate();
    }
}

void CorrectionInferenceNode::input_thread_fn() {
    auto thread_id = m_num_active_feature_threads++;

    auto fastx_reader = std::make_unique<hts_io::FastxRandomReader>(m_fastq);

    if (thread_id == 0) {
        total_reads_in_input = fastx_reader->num_entries();
    }

#ifdef DEBUG_CORRECT_PRINT_WINDOW_SIZES_TO_FILE
    std::ofstream ofs_lens("wfs_lengths." + std::to_string(thread_id) + ".txt");
    std::ofstream ofs_bed("wfs_windows." + std::to_string(thread_id) + ".bed");
#endif
#ifdef DEBUG_CORRECT_PRINT_WINDOW_INFO_TO_FILE
    std::ofstream ofs_windows("wfs_windows." + std::to_string(thread_id) + ".txt");
#endif

    Message message;
    while (get_input_message(message)) {
        if (std::holds_alternative<CorrectionAlignments>(message)) {
            utils::ScopedProfileRange spr("input_loop", 1);

            auto alignments = std::get<CorrectionAlignments>(std::move(message));
            auto tname = alignments.read_name;

            if (alignments.overlaps.empty()) {
                continue;
            }

            // If debug targets are given, skip any target that doesn't match.
            if (!std::empty(m_debug_tnames) && !m_debug_tnames.count(tname)) {
                continue;
            }

            // Get the windows
            const int32_t tlen = alignments.overlaps[0].tlen;
            const size_t n_windows = (tlen + m_window_size - 1) / m_window_size;
            LOG_TRACE("num windows {} for read {}", n_windows, alignments.read_name);
            std::vector<std::vector<OverlapWindow>> windows;
            windows.resize(n_windows);

            std::vector<secondary::Interval> win_intervals;
            std::unordered_set<int> overlap_idxs;
            bool rv = false;
            if (m_legacy_windowing) {
                rv = extract_windows(windows, alignments, m_window_size);
                for (int32_t ii = 0; ii < tlen; ii += m_window_size) {
                    win_intervals.emplace_back(
                            secondary::Interval{ii, std::min(ii + m_window_size, tlen)});
                }

                // Filter the window features and get the set of unique overlaps.
                overlap_idxs = filter_features(windows, alignments);
                if (overlap_idxs.empty()) {
                    continue;
                }

            } else {
                win_intervals =
                        extract_limited_windows(windows, alignments, m_window_size,
                                                static_cast<int32_t>(m_window_size * 1.10f));
                rv = !std::empty(windows);

                // Get the set of unique useful overlaps.
                for (const auto& win : windows) {
                    for (const auto& win_ovl : win) {
                        overlap_idxs.emplace(win_ovl.overlap_idx);
                    }
                }
                if (overlap_idxs.empty()) {
                    continue;
                }
            }

            if (!rv) {
                continue;
            }

            // Populate the alignment data with only the records that are useful after TOP_K filter
            if (!populate_alignments(alignments, fastx_reader.get(), overlap_idxs)) {
                continue;
            }

            // Get the filtered features
            auto wfs = extract_features(windows, alignments);

#ifdef DEBUG_CORRECT_PRINT_WINDOW_SIZES_TO_FILE
            for (const auto& wf : wfs) {
                ofs_lens << wf.read_name << '\t';
                polisher::print_tensor_shape(ofs_lens, wf.bases, "\t");
                ofs_lens << '\t' << alignments.overlaps[0].tlen << '\n';
            }
            for (size_t ii = 0; ii < std::size(win_intervals); ++ii) {
                const auto& interval = win_intervals[ii];
                ofs_bed << alignments.read_name << '\t' << interval.start << '\t' << interval.end
                        << '\t' << "win-" << ii << '\n';
            }
#endif
#ifdef DEBUG_CORRECT_PRINT_WINDOW_INFO_TO_FILE
            for (size_t ovl_idx = 0; ovl_idx < std::size(alignments.overlaps); ++ovl_idx) {
                const auto& ovl = alignments.overlaps[ovl_idx];
                ofs_windows << "[ovl_idx = " << ovl_idx << "] ";  // << ovl << '\n';
                utils::serialize_to_paf(ofs_windows, alignments.qnames[ovl_idx],
                                        alignments.read_name, ovl, 0, 0, 0, {});
                ofs_windows << '\n';
            }
            for (const auto& wf : wfs) {
                ofs_windows << "[window] tname = " << wf.read_name << '\t' << wf.window_idx << '\t';
                polisher::print_tensor_shape(ofs_windows, wf.bases, "\t");
                ofs_windows << "\t" << alignments.overlaps[0].tlen << "\twf = {" << wf << "}\n";
                for (size_t ii = 0; ii < std::size(windows[wf.window_idx]); ++ii) {
                    const auto& w = windows[wf.window_idx][ii];
                    ofs_windows << "    [final win i = " << ii
                                << "] qname = " << alignments.qnames[w.overlap_idx] << ", win = {"
                                << w << "}, overlap = {" << alignments.overlaps[w.overlap_idx]
                                << "}\n";
                }
                {
                    int32_t ii = 0;
                    for (const int32_t idx : overlap_idxs) {
                        ofs_windows << "    [useful ovl ii = " << ii << "] idx = " << idx
                                    << ", qname = " << alignments.qnames[idx] << '\n';
                        ++ii;
                    }
                }
            }
            ofs_windows << "-------------------\n";
#endif

            std::vector<std::string> corrected_seqs;
            corrected_seqs.resize(wfs.size());

            // Move windows that don't need inferring into an output
            // vector for later use.
            std::vector<WindowFeatures> features_to_infer;
            for (size_t w = 0; w < wfs.size(); w++) {
                if (wfs[w].n_alns > 1 && wfs[w].supported.size() > 0) {
                    features_to_infer.push_back(std::move(wfs[w]));
                } else {
                    corrected_seqs[w] = decode_window(wfs[w]);
                }
            }
            if (features_to_infer.empty()) {
                num_early_reads++;
                concat_features_and_send(corrected_seqs, tname);
            } else {
                std::lock_guard<std::mutex> lock(m_features_mutex);
                if (m_features_by_id.find(tname) == m_features_by_id.end()) {
                    m_features_by_id.insert({tname, std::move(corrected_seqs)});
                    m_pending_features_by_id.insert({tname, (int)features_to_infer.size()});
                } else {
                    spdlog::error("Features for {} already exist! Skipping.", tname);
                    continue;
                }
            }
            // Push the ones that need inference to another thread.
            for (auto& wf : features_to_infer) {
                LOG_TRACE("Pushing window idx {} to features queue", wf.window_idx);
                m_features_queue.try_push(std::move(wf));
            }
            num_reads++;

            // TODO: Remove this and move to ProgressTracker
            if (num_reads.load() % 10000 == 0) {
                spdlog::debug("Sent {} reads to inference, decoded {} reads early.",
                              num_reads.load(), num_early_reads.load());
            }
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }

    auto remaining_threads = --m_num_active_feature_threads;
    if (remaining_threads == 0) {
        m_features_queue.terminate();
    }
}

CorrectionInferenceNode::CorrectionInferenceNode(
        const std::string& fastq,
        int threads,
        const std::string& device,
        int infer_threads,
        const int batch_size,
        const std::filesystem::path& model_dir,
        const bool legacy_windowing,
        const std::unordered_set<std::string>& debug_tnames)
        : MessageSink(1000, threads),
          m_fastq(fastq),
          m_model_config(parse_model_config(model_dir / "config.toml")),
          m_features_queue(1000),
          m_inferred_features_queue(500),
          m_bases_manager(batch_size),
          m_quals_manager(batch_size),
          m_legacy_windowing(legacy_windowing),
          m_debug_tnames(debug_tnames) {
    m_window_size = m_model_config.window_size;

    if (m_window_size <= 0) {
        throw std::runtime_error{
                "Window size specified in the model config needs to be >= 0! Given: " +
                std::to_string(m_window_size)};
    }

    std::vector<std::string> devices;
    if (device == "cpu") {
        infer_threads = 1;
        devices.push_back(device);
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(device, "cuda")) {
        devices = dorado::utils::parse_cuda_device_string(device);
        if (devices.empty()) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + device);
    }
#endif
    for (size_t d = 0; d < devices.size(); d++) {
        const auto& dev = devices[d];
        const float batch_factor = (utils::starts_with(device, "cuda")) ? 0.4f : 0.8f;
        for (int i = 0; i < infer_threads; i++) {
            int device_batch_size = batch_size;
            if (batch_size == 0) {
                device_batch_size = calculate_batch_size(dev, batch_factor);
                if (device_batch_size == 0) {
                    throw std::runtime_error("Insufficient memory to run inference on " + dev);
                }
            }
            spdlog::info("Using batch size {} on device {} in inference thread {}.",
                         device_batch_size, dev, i);
            m_infer_threads.push_back(std::thread(&CorrectionInferenceNode::infer_fn, this, dev,
                                                  (int)d, device_batch_size));
        }
    }
    for (int i = 0; i < 4; i++) {
        m_decode_threads.push_back(std::thread(&CorrectionInferenceNode::decode_fn, this));
    }
    // Create index for fastq file.
    char* idx_name = fai_path(fastq.c_str());
    spdlog::debug("Looking for idx {}", idx_name);
    if (idx_name && !std::filesystem::exists(idx_name)) {
        if (fai_build(fastq.c_str()) != 0) {
            spdlog::error("Failed to build index for file {}", fastq);
            throw std::runtime_error{"Failed to build index for file " + fastq + "."};
        }
        spdlog::debug("Created fastq index.");
    }
    hts_free(idx_name);
}

void CorrectionInferenceNode::terminate(const FlushOptions&) {
    stop_input_processing();
    for (auto& infer_thread : m_infer_threads) {
        infer_thread.join();
    }
    m_infer_threads.clear();
    for (auto& decode_thread : m_decode_threads) {
        decode_thread.join();
    }
    m_decode_threads.clear();
}

stats::NamedStats CorrectionInferenceNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["num_reads_corrected"] = double(num_reads.load());
    stats["total_reads_in_input"] = total_reads_in_input;
    return stats;
}

}  // namespace dorado
