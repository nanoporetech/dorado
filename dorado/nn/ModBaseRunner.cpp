#include "ModBaseRunner.h"

#include "RemoraModel.h"
#include "modbase/remora_scaler.h"
#include "modbase/remora_utils.h"
#include "utils/base_mod_utils.h"
#include "utils/tensor_utils.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <nvtx3/nvtx3.hpp>
#include <toml.hpp>
#include <torch/torch.h>

#include <chrono>

using namespace std::chrono_literals;

namespace dorado {

void ModBaseParams::parse(std::filesystem::path const& model_path, bool all_members) {
    auto config = toml::parse(model_path / "config.toml");
    const auto& params = toml::find(config, "modbases");
    motif = toml::find<std::string>(params, "motif");
    motif_offset = toml::find<int>(params, "motif_offset");

    mod_bases = toml::find<std::string>(params, "mod_bases");
    for (size_t i = 0; i < mod_bases.size(); ++i) {
        mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    if (!all_members) {
        return;
    }

    base_mod_count = mod_long_names.size();

    context_before = toml::find<int>(params, "chunk_context_0");
    context_after = toml::find<int>(params, "chunk_context_1");
    bases_before = toml::find<int>(params, "kmer_context_bases_0");
    bases_after = toml::find<int>(params, "kmer_context_bases_1");
    offset = toml::find<int>(params, "offset");

    try {
        // these may not exist if we convert older models
        const auto& refinement_params = toml::find(config, "refinement");
        refine_do_rough_rescale =
                (toml::find<int>(refinement_params, "refine_do_rough_rescale") == 1);
        if (refine_do_rough_rescale) {
            refine_kmer_center_idx = toml::find<int>(refinement_params, "refine_kmer_center_idx");

            auto kmer_levels_tensor =
                    utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0].contiguous();
            std::copy(kmer_levels_tensor.data_ptr<float>(),
                      kmer_levels_tensor.data_ptr<float>() + kmer_levels_tensor.numel(),
                      std::back_inserter(refine_kmer_levels));
            refine_kmer_len = static_cast<size_t>(
                    std::round(std::log(refine_kmer_levels.size()) / std::log(4)));
        }

    } catch (const std::out_of_range& ex) {
        // no refinement parameters
        refine_do_rough_rescale = false;
    }
}

class ModBaseCaller {
public:
    struct ModBaseTask {
        ModBaseTask(torch::Tensor input_sigs_, torch::Tensor input_seqs_, int num_chunks_)
                : input_sigs(input_sigs_), input_seqs(input_seqs_), num_chunks(num_chunks_) {}
        torch::Tensor input_sigs;
        torch::Tensor input_seqs;
        std::mutex mut;
        std::condition_variable cv;
        torch::Tensor out;
        bool done{false};
        int num_chunks;
    };

    struct ModBaseData {
        torch::nn::ModuleHolder<torch::nn::AnyModule> module_holder{nullptr};
        std::unique_ptr<RemoraScaler> scaler{nullptr};
        ModBaseParams params{};
        std::deque<ModBaseTask*> input_queue;
        std::mutex input_lock;
        std::condition_variable input_cv;
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        c10::optional<c10::Stream> stream;
#endif
        int batch_size = 0;

        torch::Tensor scale_signal(torch::Tensor signal,
                                   const std::vector<int>& seq_ints,
                                   const std::vector<uint64_t>& seq_to_sig_map) const {
            NVTX3_FUNC_RANGE();
            if (!scaler) {
                return signal;
            }

            auto levels = scaler->extract_levels(seq_ints);

            // generate the signal values at the centre of each base, create the nx5% quantiles (sorted)
            // and perform a linear regression against the expected kmer levels to generate a new shift and scale
            auto [offset, scale] = scaler->rescale(signal, seq_to_sig_map, levels);
            auto scaled_signal = signal * scale + offset;
            return scaled_signal;
        }

        std::vector<size_t> get_motif_hits(const std::string& seq) const {
            NVTX3_FUNC_RANGE();
            std::vector<size_t> context_hits;
            const auto& motif = params.motif;
            const auto motif_offset = params.motif_offset;
            size_t kmer_len = motif.size();
            size_t search_pos = 0;
            while (search_pos < seq.size() - kmer_len + 1) {
                search_pos = seq.find(motif, search_pos);
                if (search_pos != std::string::npos) {
                    context_hits.push_back(search_pos + motif_offset);
                    ++search_pos;
                }
            }
            return context_hits;
        }
    };

    ModBaseCaller(const std::vector<std::filesystem::path>& model_paths,
                  int batch_size,
                  const std::string& device) {
        // no metal implementation yet, force to cpu
        if (device == "metal" || device == "cpu") {
            // no slow_conv2d_cpu for type Half, need to use float32
            m_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        } else {
            m_options = torch::TensorOptions().device(device).dtype(torch::kFloat16);
        }

        for (size_t model_id = 0; model_id < model_paths.size(); ++model_id) {
            const auto& model_path = model_paths[model_id];
            auto caller_data = std::make_unique<ModBaseData>();
#if DORADO_GPU_BUILD && !defined(__APPLE__)
            if (m_options.device().is_cuda()) {
                caller_data->stream =
                        c10::cuda::getStreamFromPool(false, m_options.device().index());
            }
#endif
            caller_data->module_holder = load_remora_model(model_path, m_options);
            caller_data->params.parse(model_path);
            caller_data->batch_size = batch_size;

            auto sig_len = static_cast<int64_t>(caller_data->params.context_before +
                                                caller_data->params.context_after);
            auto kmer_len = caller_data->params.bases_after + caller_data->params.bases_before + 1;

            // Warmup
            auto input_sigs = torch::empty({batch_size, 1, sig_len}, m_options);
            auto input_seqs = torch::empty({batch_size, RemoraUtils::NUM_BASES * kmer_len, sig_len},
                                           m_options);
            caller_data->module_holder->forward(input_sigs, input_seqs);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
            if (m_options.device().is_cuda()) {
                torch::cuda::synchronize(m_options.device().index());
            }
#endif
            if (caller_data->params.refine_do_rough_rescale) {
                caller_data->scaler = std::make_unique<RemoraScaler>(
                        caller_data->params.refine_kmer_levels, caller_data->params.refine_kmer_len,
                        caller_data->params.refine_kmer_center_idx);
            }

            m_caller_data.push_back(std::move(caller_data));
            m_task_threads.push_back(std::make_unique<std::thread>(
                    &ModBaseCaller::modbase_task_thread_fn, this, model_id));
        }
    }

    ~ModBaseCaller() {
        m_terminate.store(true);
        for (auto& caller_data : m_caller_data) {
            caller_data->input_cv.notify_one();
        }

        for (auto& task_thread : m_task_threads) {
            task_thread->join();
        }
    }

    torch::Tensor call_chunks(size_t model_id,
                              torch::Tensor& input_sigs,
                              torch::Tensor& input_seqs,
                              int num_chunks) {
        NVTX3_FUNC_RANGE();
        auto& caller_data = m_caller_data[model_id];

#if DORADO_GPU_BUILD && !defined(__APPLE__)
        c10::cuda::OptionalCUDAStreamGuard stream_guard(caller_data->stream);
#endif
        ModBaseTask task(input_sigs.to(m_options.device()), input_seqs.to(m_options.device()),
                         num_chunks);
        {
            std::lock_guard<std::mutex> lock(caller_data->input_lock);
            caller_data->input_queue.push_front(&task);
        }
        caller_data->input_cv.notify_one();

        std::unique_lock lock(task.mut);
        while (!task.done) {
            task.cv.wait(lock);
        }

        return task.out;
    }

    void modbase_task_thread_fn(size_t model_id) {
        NVTX3_FUNC_RANGE();
        auto& caller_data = m_caller_data[model_id];
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        const bool has_stream = caller_data->stream.has_value();
#endif
        while (true) {
            torch::InferenceMode guard;
#if DORADO_GPU_BUILD && !defined(__APPLE__)
            // If caller_data->stream is set, sets the current stream to caller_data->stream, and the current device to
            // the device associated with the stream. Resets both to their prior state on destruction
            c10::cuda::OptionalCUDAStreamGuard stream_guard(caller_data->stream);
#endif
            std::unique_lock<std::mutex> input_lock(caller_data->input_lock);
            while (caller_data->input_queue.empty() && !m_terminate.load()) {
                caller_data->input_cv.wait_for(input_lock, 100ms);
            }

            if (caller_data->input_queue.empty() && m_terminate.load()) {
                return;
            }

            ModBaseTask* task = caller_data->input_queue.back();
            caller_data->input_queue.pop_back();
            input_lock.unlock();

            std::unique_lock<std::mutex> task_lock(task->mut);
            auto scores = caller_data->module_holder->forward(task->input_sigs, task->input_seqs);
            task->out = scores.to(torch::kCPU);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
            if (has_stream) {
                caller_data->stream->synchronize();
            }
#endif
            task->done = true;
            task->cv.notify_one();
            task_lock.unlock();
        }
    }

    torch::TensorOptions m_options;
    std::atomic<bool> m_terminate{false};
    std::vector<std::unique_ptr<ModBaseData>> m_caller_data;
    std::vector<std::unique_ptr<std::thread>> m_task_threads;
};

std::shared_ptr<ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device) {
    return std::make_shared<ModBaseCaller>(model_paths, batch_size, device);
}

ModBaseRunner::ModBaseRunner(std::shared_ptr<ModBaseCaller> caller) : m_caller(std::move(caller)) {
    auto opts = torch::TensorOptions()
                        .device(torch::kCPU)
                        .pinned_memory(m_caller->m_options.device().is_cuda())
                        .dtype(m_caller->m_options.dtype());

    for (auto& caller_data : m_caller->m_caller_data) {
        auto sig_len = static_cast<int64_t>(caller_data->params.context_before +
                                            caller_data->params.context_after);
        auto kmer_len = caller_data->params.bases_after + caller_data->params.bases_before + 1;
        m_input_sigs.push_back(torch::empty({caller_data->batch_size, 1, sig_len}, opts));
        m_input_seqs.push_back(torch::empty(
                {caller_data->batch_size, RemoraUtils::NUM_BASES * kmer_len, sig_len}, opts));
    }
}

void ModBaseRunner::accept_chunk(int model_id,
                                 int chunk_idx,
                                 const torch::Tensor& signal,
                                 const std::vector<float>& kmers) {
    // As usual, avoid torch indexing because it is glacially slow.
    // GPU base calling uses float16 signals and input tensors, but float32
    // sequence encodings.
    // CPU base calling uses float16 signals, float32 input tensors, and
    // float32 sequence encodings.

    auto& input_sigs = m_input_sigs[model_id];
    auto& input_seqs = m_input_seqs[model_id];
    assert(signal.size(0) == input_sigs.size(2));

    const auto sig_len = signal.size(0);
    dorado::utils::copy_tensor_elems(input_sigs, chunk_idx * sig_len, signal, 0, sig_len);

    const auto kmer_elem_count = input_seqs.size(1) * input_seqs.size(2);
    if (input_seqs.dtype() == torch::kFloat16) {
        // float32 kmer encoding -> float16 kmer encoding input
        using InputType = c10::Half;
        InputType* const input_seqs_ptr = input_seqs.data_ptr<InputType>();
        dorado::utils::convert_f32_to_f16(&input_seqs_ptr[chunk_idx * kmer_elem_count],
                                          kmers.data(), kmer_elem_count);
    } else if (input_seqs.dtype() == torch::kFloat32) {
        // float32 kmer encoding -> float32 kmer encoding input
        using InputType = float;
        InputType* const input_seqs_ptr = input_seqs.data_ptr<InputType>();
        std::memcpy(&input_seqs_ptr[chunk_idx * kmer_elem_count], kmers.data(),
                    kmer_elem_count * sizeof(InputType));
    } else {
        throw std::runtime_error("Unsupported input dtype");
    }
}

torch::Tensor ModBaseRunner::call_chunks(int model_id, int num_chunks) {
    return m_caller->call_chunks(model_id, m_input_sigs[model_id], m_input_seqs[model_id],
                                 num_chunks);
}

torch::Tensor ModBaseRunner::scale_signal(size_t caller_id,
                                          torch::Tensor signal,
                                          const std::vector<int>& seq_ints,
                                          const std::vector<uint64_t>& seq_to_sig_map) const {
    return m_caller->m_caller_data[caller_id]->scale_signal(signal, seq_ints, seq_to_sig_map);
}

std::vector<size_t> ModBaseRunner::get_motif_hits(size_t caller_id, const std::string& seq) const {
    return m_caller->m_caller_data[caller_id]->get_motif_hits(seq);
}

ModBaseParams& ModBaseRunner::caller_params(size_t caller_id) const {
    return m_caller->m_caller_data[caller_id]->params;
}

size_t ModBaseRunner::num_callers() const { return m_caller->m_caller_data.size(); }

}  // namespace dorado
