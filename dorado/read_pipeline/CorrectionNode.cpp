#include "CorrectionNode.h"

#include "utils/bam_utils.h"
#include "utils/gpu_profiling.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/types.h"
#if DORADO_CUDA_BUILD
#include "utils/cuda_utils.h"
#endif
#include "correct/conversions.h"
#include "correct/decode.h"
#include "correct/features.h"
#include "correct/infer.h"
#include "correct/windows.h"
#include "hts_io/FastxRandomReader.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>
#endif
#include <htslib/faidx.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/nn/utils/rnn.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dorado::correction;

namespace {

dorado::BamPtr create_bam_record(const std::string& read_id, const std::string& seq) {
    bam1_t* rec = bam_init1();
    bam_set1(rec, read_id.length(), read_id.c_str(), 4 /*flag*/, -1 /*tid*/, -1 /*pos*/, 0 /*mapq*/,
             0 /*n_cigar*/, nullptr /*cigar*/, -1 /*mtid*/, -1 /*mpos*/, 0 /*isize*/, seq.size(),
             seq.data(), nullptr, 0);
    return dorado::BamPtr(rec);
}

void populate_alignments(dorado::CorrectionAlignments& alignments,
                         dorado::hts_io::FastxRandomReader* reader) {
    const auto& tname = alignments.read_name;
    alignments.read_seq = reader->fetch_seq(tname);
    alignments.read_qual = reader->fetch_qual(tname);
    int tlen = (int)alignments.read_seq.length();
    auto num_qnames = alignments.qnames.size();
    alignments.seqs.resize(num_qnames);
    alignments.quals.resize(num_qnames);
    for (size_t i = 0; i < num_qnames; i++) {
        const std::string& qname = alignments.qnames[i];
        alignments.seqs[i] = reader->fetch_seq(qname);
        alignments.quals[i] = reader->fetch_qual(qname);
        alignments.overlaps[i].tlen = tlen;
    }
}

}  // namespace

namespace dorado {

void CorrectionNode::decode_fn() {
    spdlog::info("Starting decode thread!");
    m_num_active_decode_threads++;

    WindowFeatures item;
    while (m_inferred_features_queue.try_pop(item) != utils::AsyncQueueStatus::Terminate) {
        //spdlog::info("Popped inferred feature for {}", item.window_idx);
        utils::ScopedProfileRange spr("decode_loop", 1);
        std::vector<WindowFeatures> to_decode;
        {
            std::lock_guard<std::mutex> lock(m_features_mutex);
            auto pos = item.window_idx;
            auto read_name = item.read_name;
            auto find_iter = m_features_by_id.find(read_name);
            auto& output_features = find_iter->second;
            output_features[pos] = std::move(item);
            //spdlog::info("replaced window in position {}", pos);
            auto& pending = m_pending_features_by_id.find(read_name)->second;
            pending--;
            if (pending == 0) {
                //spdlog::info("Got all features!");
                // Got all features!
                to_decode = std::move(output_features);
                m_features_by_id.erase(read_name);
                m_pending_features_by_id.erase(read_name);
            }
        }

        if (!to_decode.empty()) {
            auto t0 = std::chrono::high_resolution_clock::now();
            const std::string& read_name = to_decode[0].read_name;
            //spdlog::info("decoding window now for {}", read_name);
            auto corrected_seqs = decode_windows(to_decode);
            if (corrected_seqs.size() == 1) {
                auto rec = create_bam_record(read_name, corrected_seqs[0]);
                send_message_to_sink(std::move(rec));
            } else {
                for (size_t s = 0; s < corrected_seqs.size(); s++) {
                    const std::string new_name = read_name + ":" + std::to_string(s);
                    auto rec = create_bam_record(new_name, corrected_seqs[s]);
                    send_message_to_sink(std::move(rec));
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = t1 - t0;
            {
                std::lock_guard<std::mutex> lock(decodeMutex);
                decodeDuration += duration;
            }

            //for (auto& wf : to_decode) {
            //    if (wf.n_alns > 1) {
            //        int* bases_ptr = wf.bases.data_ptr<int>();
            //        float* quals_ptr = wf.quals.data_ptr<float>();
            //        m_bases_manager.return_ptr(bases_ptr);
            //        m_quals_manager.return_ptr(quals_ptr);
            //    }
            //}
        }
    }

    m_num_active_decode_threads--;
}

void CorrectionNode::infer_fn(const std::string& device_str, int mtx_idx) {
    spdlog::info("Starting process thread!");

    m_num_active_infer_threads++;

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/home/OXFORDNANOLABS/jdaw/github/haec-BigBird/ont-model.pt");
    } catch (const c10::Error& e) {
        spdlog::error("Error loading model");
        throw std::runtime_error("");
    }

    spdlog::debug("Loaded model!");
    //const auto device = torch::kCPU;
    torch::Device device = torch::Device(device_str);
    torch::NoGradGuard no_grad;
    module.to(device);
    module.eval();

#if DORADO_CUDA_BUILD
    auto stream = c10::cuda::getStreamFromPool(false, device.index());

    torch::DeviceGuard device_guard(device);
    torch::StreamGuard stream_guard(stream);
#endif

    bool first_inference = true;

    std::vector<torch::Tensor> bases_batch;
    std::vector<torch::Tensor> quals_batch;
    std::vector<int> lengths;
    std::vector<int64_t> sizes;
    std::vector<torch::Tensor> indices_batch;
    std::vector<WindowFeatures> wfs;
    // If there are any windows > 5120, then reduce batch size by 1
    int remaining_batch_slots = m_batch_size;

    auto decode_preds = [](const torch::Tensor& preds) {
        std::vector<char> bases;
        bases.reserve(preds.sizes()[0]);
        static std::array<char, 5> decoder = {'A', 'C', 'G', 'T', '*'};
        for (int i = 0; i < preds.sizes()[0]; i++) {
            auto base_idx = preds[i].item<int>();
            bases.push_back(decoder[base_idx]);
            //spdlog::info("{} decoded to {}", i, bases.back());
        }
        return bases;
    };

    auto batch_infer = [&]() {
        utils::ScopedProfileRange infer("infer", 1);
        if (first_inference) {
            spdlog::info("Calling inference");
        }
        // Run inference on batch
        auto t0 = std::chrono::high_resolution_clock::now();
        auto length_tensor =
                torch::from_blob(lengths.data(), {(int)lengths.size()},
                                 torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        auto batched_bases =
                collate<int>(bases_batch, (int)11, torch::kInt32, m_bases_manager.get_next_ptr());
        auto batched_quals =
                collate<float>(quals_batch, 0.f, torch::kFloat32, m_quals_manager.get_next_ptr());
        //print_size(batched_bases, "batched_bases");
        //print_size(batched_quals, "batched_quals");
        //print_size(length_tensor, "length_tensor");

        auto t1 = std::chrono::high_resolution_clock::now();

        std::unique_lock<std::mutex> lock(m_gpu_mutexes[mtx_idx]);
        std::vector<torch::jit::IValue> inputs;
        {
            utils::ScopedProfileRange move_to_device("move_to_device", 1);
            inputs.push_back(batched_bases.to(device));
            inputs.push_back(batched_quals.to(device));
            inputs.push_back(length_tensor.to(device));
            std::for_each(indices_batch.begin(), indices_batch.end(),
                          [device](torch::Tensor& t) { t.to(device); });
            inputs.push_back(indices_batch);
        }
        auto t2 = std::chrono::high_resolution_clock::now();

#if 1
        auto output = module.forward(inputs);
        if (output.isTuple()) {
            auto base_logits = output.toTuple()->elements()[1].toTensor();
            //print_size(base_logits, "base_logits");
            auto preds = base_logits.argmax(1, false).to(torch::kCPU);
            lock.unlock();
            auto t3 = std::chrono::high_resolution_clock::now();
            //print_size(preds, "preds");
            auto split_preds = preds.split_with_sizes(sizes);
            //spdlog::info("split preds size {}", split_preds.size());
            for (size_t w = 0; w < split_preds.size(); w++) {
                auto decoded_output = decode_preds(split_preds[w]);
                //spdlog::info("decoded output size {}", decoded_output.size());
                wfs[w].inferred_bases = decoded_output;
                //spdlog::info("stored output");
            }
            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> collate_time = (t1 - t0);
            std::chrono::duration<double> move_to_device = (t2 - t1);
            std::chrono::duration<double> forward = (t3 - t2);
            std::chrono::duration<double> dec = (t4 - t3);
            if (first_inference) {
                spdlog::info("collate {} move_to_dev {} forward {} dec {}", collate_time.count(),
                             move_to_device.count(), forward.count(), dec.count());
            }

            for (auto& wf : wfs) {
                //spdlog::info("Pushing inferred features for {} window", wf.window_idx);
                m_inferred_features_queue.try_push(std::move(wf));
            }
        }
#endif
        m_bases_manager.return_ptr(batched_bases.data_ptr<int>());
        m_quals_manager.return_ptr(batched_quals.data_ptr<float>());

        bases_batch.clear();
        quals_batch.clear();
        lengths.clear();
        sizes.clear();
        wfs.clear();
        indices_batch.clear();
        first_inference = false;
        remaining_batch_slots = m_batch_size;
        //spdlog::info("reset actual bs {}", remaining_batch_slots);
    };

    WindowFeatures item;
    while (m_features_queue.try_pop(item) != utils::AsyncQueueStatus::Terminate) {
        utils::ScopedProfileRange spr("collect_features", 1);
        //spdlog::info("remaining batch spots {}", remaining_batch_slots);
        int required_batch_slots = (item.bases.sizes()[1] / 5120) + 1;
        if (required_batch_slots > remaining_batch_slots) {
            batch_infer();
        }
        wfs.push_back(std::move(item));
        auto& wf = wfs.back();

        //spdlog::info("Popped window idx {}", wf.window_idx);
        //for(int i = 0; i < wf.bases.sizes()[0]; i++) {
        //    spdlog::info("target row before inference pos {} base {}", i, wf.bases[i][0].item<int>());
        //}
        //print_size(wf.bases, "popped from features queue");
        auto b = wf.bases.transpose(0, 1);
        //auto q = 2.f * (wf.quals.transpose(0, 1) - 33.f) / (126.f - 33.f) - 1.f;
        auto q = wf.quals.transpose(0, 1);
        //q.sub_(33.f).div_(126.f - 33.f).mul_(2.f).sub_(1.f);
        //auto q = wf.quals;

        bases_batch.push_back(b);
        quals_batch.push_back(q);
        lengths.push_back(wf.length);
        sizes.push_back(wf.length);
        indices_batch.push_back(wf.indices);
        remaining_batch_slots -= required_batch_slots;
        //print_size(wf.bases, "bases");
        //print_size(wf.quals, "quals");
        //print_size(wf.indices, "indices");
        //spdlog::info("length {}", wf.length);

        //spdlog::info("bases max {} min {} sum {}", wf.bases.max().item<uint8_t>(), wf.bases.min().item<uint8_t>(), wf.bases.sum().item<int>());
        //spdlog::info("quals max {} min {} sum {}", wf.quals.max().item<float>(), wf.quals.min().item<float>(), wf.quals.sum().item<float>());
    }

    if (bases_batch.size() > 0) {
        batch_infer();
    }

    m_num_active_infer_threads--;
    if (m_num_active_infer_threads.load() == 0) {
        m_inferred_features_queue.terminate();
    }
}

void CorrectionNode::input_thread_fn() {
    Message message;

    m_num_active_feature_threads++;

    auto fastx_reader = std::make_unique<hts_io::FastxRandomReader>(m_fastq);
    while (get_input_message(message)) {
        if (std::holds_alternative<CorrectionAlignments>(message)) {
            auto alignments = std::get<CorrectionAlignments>(std::move(message));
            utils::ScopedProfileRange spr("input_loop", 1);
            populate_alignments(alignments, fastx_reader.get());
            //if (alignments.read_name == "a0ce7c4c-28d3-46cd-aca7-9a203b87ffc6") {
            if (true) {
                //spdlog::info("Process windows for {} of length {}", alignments.read_name,
                //             alignments.read_seq.length());
                size_t n_windows =
                        (alignments.read_seq.length() + m_window_size - 1) / m_window_size;
                //spdlog::info("num windows {}", n_windows);
                std::vector<std::vector<OverlapWindow>> windows;
                windows.resize(n_windows);
                auto t0 = std::chrono::high_resolution_clock::now();
                // Get the windows
                extract_windows(windows, alignments, m_window_size);
                //int o = 0;
                //for (auto& ovlp_windows : windows) {
                //    spdlog::info("{} ovlps in window {}", ovlp_windows.size(), o++);
                //}
                auto t1 = std::chrono::high_resolution_clock::now();
                // Get the features
                auto wfs = extract_features(windows, alignments, m_window_size);
                auto t2 = std::chrono::high_resolution_clock::now();
                std::vector<WindowFeatures> features_to_infer;

                // Move features that don't need inferring into an output
                // vector for later use.
                for (size_t w = 0; w < wfs.size(); w++) {
                    if (wfs[w].n_alns > 1 && wfs[w].supported.size() > 0) {
                        features_to_infer.push_back(std::move(wfs[w]));
                    }
                }
                //spdlog::info("Have {} pending features {} done features", features_to_infer.size(), output.size());
                {
                    std::lock_guard<std::mutex> lock(m_features_mutex);
                    m_features_by_id.insert({alignments.read_name, std::move(wfs)});
                    m_pending_features_by_id.insert(
                            {alignments.read_name, (int)features_to_infer.size()});
                }
                // Push the ones that need inference to another thread.
                for (auto& wf : features_to_infer) {
                    //spdlog::info("Pushing window idx {} to features queue", wf.window_idx);
                    m_features_queue.try_push(std::move(wf));
                }
                {
                    std::chrono::duration<double> duration = t1 - t0;
                    std::lock_guard<std::mutex> lock(ewMutex);
                    extractWindowsDuration += duration;
                }
                {
                    std::chrono::duration<double> duration = t2 - t1;
                    std::lock_guard<std::mutex> lock(efMutex);
                    extractFeaturesDuration += duration;
                }
            }
            num_reads++;

            if (num_reads.load() % 10000 == 0) {
                spdlog::info("Processed {} reads", num_reads.load());
            }
        } else {
            send_message_to_sink(std::move(message));
            continue;
        }
    }

    m_num_active_feature_threads--;
    if (m_num_active_feature_threads.load() == 0) {
        m_features_queue.terminate();
    }
}

CorrectionNode::CorrectionNode(const std::string& fastq,
                               int threads,
                               const std::string& device,
                               int infer_threads,
                               int batch_size)
        : MessageSink(1000, threads),
          m_fastq(fastq),
          m_batch_size(batch_size),
          m_features_queue(1024),
          m_inferred_features_queue(512),
          m_bases_manager(batch_size, gen_base_encoding()['.']),
          m_quals_manager(batch_size, (float)'!') {
    std::vector<std::string> devices;
    if (device == "cpu") {
        infer_threads = 1;
        devices.push_back(device);
    }
#ifdef __APPLE__
    else if (device == "mps") {
        devices.push_back("mps");
    }
#endif
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(device, "cuda")) {
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA backend not available. Choose another one.");
        }
        devices = dorado::utils::parse_cuda_device_string(device);
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + device);
    }
#endif
    for (size_t d = 0; d < devices.size(); d++) {
        const auto& dev = devices[d];
        for (int i = 0; i < infer_threads; i++) {
            m_infer_threads.push_back(
                    std::make_unique<std::thread>(&CorrectionNode::infer_fn, this, dev, (int)d));
        }
    }
    for (int i = 0; i < 4; i++) {
        m_decode_threads.push_back(std::make_unique<std::thread>(&CorrectionNode::decode_fn, this));
    }
    // Create index for fastq file.
    char* idx_name = fai_path(fastq.c_str());
    spdlog::info("Looking for idx {}", idx_name);
    if (!std::filesystem::exists(idx_name)) {
        if (fai_build(fastq.c_str()) != 0) {
            spdlog::error("Failed to build index for file {}", fastq);
            throw std::runtime_error("");
        }
        spdlog::info("Created fastq index.");
    }
    free(idx_name);
    start_input_processing(&CorrectionNode::input_thread_fn, this);
}

void CorrectionNode::terminate(const FlushOptions&) {
    stop_input_processing();
    for (auto& infer_thread : m_infer_threads) {
        if (infer_thread->joinable()) {
            infer_thread->join();
        }
    }
    for (auto& decode_thread : m_decode_threads) {
        if (decode_thread->joinable()) {
            decode_thread->join();
        }
    }
}

stats::NamedStats CorrectionNode::sample_stats() const {
    stats::NamedStats stats = stats::from_obj(m_work_queue);
    stats["reads_processed"] = double(num_reads.load());
    return stats;
}

}  // namespace dorado
