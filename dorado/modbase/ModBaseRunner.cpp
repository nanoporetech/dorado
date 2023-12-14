#include "ModBaseRunner.h"

#include "ModBaseModel.h"
#include "ModBaseModelConfig.h"
#include "ModbaseScaler.h"
#include "MotifMatcher.h"
#include "utils/sequence_utils.h"
#include "utils/stats.h"
#include "utils/tensor_utils.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <chrono>

using namespace std::chrono_literals;

namespace dorado::modbase {

class ModBaseCaller {
public:
    struct ModBaseTask {
        ModBaseTask(at::Tensor input_sigs_, at::Tensor input_seqs_, int num_chunks_)
                : input_sigs(input_sigs_), input_seqs(input_seqs_), num_chunks(num_chunks_) {}
        at::Tensor input_sigs;
        at::Tensor input_seqs;
        std::mutex mut;
        std::condition_variable cv;
        at::Tensor out;
        bool done{false};
        int num_chunks;
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        c10::optional<c10::Stream> stream;
#endif
    };

    struct ModBaseData {
        torch::nn::ModuleHolder<torch::nn::AnyModule> module_holder;
        std::unique_ptr<ModBaseScaler> scaler{nullptr};
        const ModBaseModelConfig params;
        const MotifMatcher matcher;
        std::deque<std::shared_ptr<ModBaseTask>> input_queue;
        std::mutex input_lock;
        std::condition_variable input_cv;
        const int batch_size;

        std::vector<size_t> get_motif_hits(const std::string& seq) const {
            return matcher.get_motif_hits(seq);
        }

        ModBaseData(const std::filesystem::path& model_path,
                    at::TensorOptions opts,
                    int batch_size_)
                : module_holder(load_modbase_model(model_path, opts)),
                  params(load_modbase_model_config(model_path)),
                  matcher(params),
                  batch_size(batch_size_) {
            if (params.refine_do_rough_rescale) {
                scaler = std::make_unique<ModBaseScaler>(params.refine_kmer_levels,
                                                         params.refine_kmer_len,
                                                         params.refine_kmer_center_idx);
            }

#if DORADO_GPU_BUILD && !defined(__APPLE__)
            if (opts.device().is_cuda()) {
                auto sig_len = static_cast<int64_t>(params.context_before + params.context_after);
                auto kmer_len = params.bases_after + params.bases_before + 1;

                // Warmup
                auto input_sigs = torch::empty({batch_size, 1, sig_len}, opts);
                auto input_seqs = torch::empty(
                        {batch_size, sig_len, utils::BaseInfo::NUM_BASES * kmer_len}, opts);
                module_holder->forward(input_sigs, input_seqs);
                torch::cuda::synchronize(opts.device().index());
            }
#endif
        }
    };

    ModBaseCaller(const std::vector<std::filesystem::path>& model_paths,
                  int batch_size,
                  const std::string& device)
            : m_num_models(model_paths.size()) {
        if (device == "cpu") {
            // no slow_conv2d_cpu for type Half, need to use float32
            m_options = at::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
        } else if (device == "metal") {
#if TORCH_VERSION_MAJOR < 2
            // no metal implementation yet, force to cpu
            auto torchMetalBackend = torch::kCPU;
            auto torchMetalDtype = torch::kFloat32;
            spdlog::debug(
                    "- no metal backend available for modified basecalling, defaulting to CPU.");
#else
            auto torchMetalBackend = torch::kMPS;
            auto torchMetalDtype = torch::kFloat16;
#endif
            m_options = at::TensorOptions().device(torchMetalBackend).dtype(torchMetalDtype);
        } else {
            m_options = at::TensorOptions().device(device).dtype(torch::kFloat16);
        }

        // Allocate enough elements up-front so that m_caller_data.push_back() doesn't reallocate while
        // other threads can be referencing elements that it's holding.
        m_caller_data.reserve(m_num_models);
        m_task_threads.reserve(m_num_models);

        for (size_t model_id = 0; model_id < m_num_models; ++model_id) {
            const auto& model_path = model_paths[model_id];

            at::InferenceMode guard;
            auto caller_data = std::make_unique<ModBaseData>(model_path, m_options, batch_size);
            m_caller_data.push_back(std::move(caller_data));
        }

        start_threads();
    }

    void start_threads() {
        for (size_t model_id = 0; model_id < m_num_models; ++model_id) {
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

    at::Tensor call_chunks(size_t model_id,
                           at::Tensor& input_sigs,
                           at::Tensor& input_seqs,
                           int num_chunks) {
        NVTX3_FUNC_RANGE();
        auto& caller_data = m_caller_data[model_id];
        auto task = std::make_shared<ModBaseTask>(input_sigs.to(m_options.device()),
                                                  input_seqs.to(m_options.device()), num_chunks);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        if (m_options.device().is_cuda()) {
            task->stream = c10::cuda::getCurrentCUDAStream(m_options.device().index());
        }
#endif
        {
            std::lock_guard<std::mutex> lock(caller_data->input_lock);
            caller_data->input_queue.push_front(task);
        }
        caller_data->input_cv.notify_one();

        std::unique_lock lock(task->mut);
        while (!task->done) {
            task->cv.wait(lock);
        }

        return task->out.to(torch::kCPU);
    }

    void modbase_task_thread_fn(size_t model_id) {
        auto& caller_data = m_caller_data[model_id];
        while (true) {
            nvtx3::scoped_range loop{"modbase_task_thread_fn"};
            at::InferenceMode guard;

            std::unique_lock<std::mutex> input_lock(caller_data->input_lock);
            while (caller_data->input_queue.empty() && !m_terminate.load()) {
                caller_data->input_cv.wait_for(input_lock, 100ms);
            }

            if (caller_data->input_queue.empty() && m_terminate.load()) {
                return;
            }

            auto task = caller_data->input_queue.back();
            caller_data->input_queue.pop_back();
            input_lock.unlock();

#if DORADO_GPU_BUILD && !defined(__APPLE__)
            // If task->stream is set, sets the current stream to task->stream, and the current device to
            // the device associated with the stream. Resets both to their prior state on destruction
            c10::cuda::OptionalCUDAStreamGuard stream_guard(task->stream);
#endif

            std::unique_lock<std::mutex> task_lock(task->mut);
            stats::Timer timer;
            task->out = caller_data->module_holder->forward(task->input_sigs, task->input_seqs);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
            if (task->stream.has_value()) {
                task->stream->synchronize();
            }
            // Only meaningful if we're syncing the stream.
            m_model_ms += timer.GetElapsedMS();
#endif
            ++m_num_batches_called;
            task->done = true;
            task_lock.unlock();
            task->cv.notify_one();
        }
    }

    void terminate() {
        m_terminate.store(true);
        for (auto& caller_data : m_caller_data) {
            caller_data->input_cv.notify_one();
        }
        for (auto& task_thread : m_task_threads) {
            task_thread->join();
        }
        m_task_threads.clear();
    }

    void restart() {
        if (m_terminate.load()) {
            m_terminate.store(false);
            start_threads();
        }
    }

    std::string get_name() const {
        return std::string("ModBaseCaller_") + m_options.device().str();
    }

    stats::NamedStats sample_stats() const {
        stats::NamedStats stats;
        stats["batches_called"] = double(m_num_batches_called);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        stats["model_ms"] = double(m_model_ms);
#endif
        return stats;
    }

    size_t m_num_models = 0;

    at::TensorOptions m_options;
    std::atomic<bool> m_terminate{false};
    std::vector<std::unique_ptr<ModBaseData>> m_caller_data;
    std::vector<std::unique_ptr<std::thread>> m_task_threads;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
};

std::shared_ptr<ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device) {
    return std::make_shared<ModBaseCaller>(model_paths, batch_size, device);
}

ModBaseRunner::ModBaseRunner(std::shared_ptr<ModBaseCaller> caller) : m_caller(std::move(caller)) {
    auto opts = at::TensorOptions()
                        .device(torch::kCPU)
                        .pinned_memory(m_caller->m_options.device().is_cuda())
                        .dtype(m_caller->m_options.dtype());

    auto seq_input_options = at::TensorOptions()
                                     .device(torch::kCPU)
                                     .pinned_memory(m_caller->m_options.device().is_cuda())
                                     .dtype(torch::kInt8);

    for (auto& caller_data : m_caller->m_caller_data) {
        auto sig_len = static_cast<int64_t>(caller_data->params.context_before +
                                            caller_data->params.context_after);
        auto kmer_len = caller_data->params.bases_after + caller_data->params.bases_before + 1;
        m_input_sigs.push_back(torch::empty({caller_data->batch_size, 1, sig_len}, opts));
        m_input_seqs.push_back(torch::empty(
                {caller_data->batch_size, sig_len, utils::BaseInfo::NUM_BASES * kmer_len},
                seq_input_options));
#if DORADO_GPU_BUILD && !defined(__APPLE__)
        if (m_caller->m_options.device().is_cuda()) {
            m_streams.push_back(
                    c10::cuda::getStreamFromPool(false, m_caller->m_options.device().index()));
        } else {
            m_streams.emplace_back();
        }
#endif
    }
}

void ModBaseRunner::accept_chunk(int model_id,
                                 int chunk_idx,
                                 const at::Tensor& signal,
                                 const std::vector<int8_t>& kmers) {
    // As usual, avoid torch indexing because it is glacially slow.
    // GPU base calling uses float16 signals and input tensors.
    // CPU base calling uses float16 signals, float32 input tensors.
    // Both versions take int8 sequence encodings.

    auto& input_sigs = m_input_sigs[model_id];
    auto& input_seqs = m_input_seqs[model_id];
    assert(signal.size(0) == input_sigs.size(2));

    const auto sig_len = signal.size(0);
    dorado::utils::copy_tensor_elems(input_sigs, chunk_idx * sig_len, signal, 0, sig_len);

    const auto kmer_elem_count = input_seqs.size(1) * input_seqs.size(2);
    if (input_seqs.dtype() != torch::kInt8) {
        throw std::runtime_error("Unsupported input dtype");
    }
    using SeqInputType = int8_t;
    SeqInputType* const input_seqs_ptr = input_seqs.data_ptr<SeqInputType>();
    std::memcpy(&input_seqs_ptr[chunk_idx * kmer_elem_count], kmers.data(),
                kmer_elem_count * sizeof(SeqInputType));
}

at::Tensor ModBaseRunner::call_chunks(int model_id, int num_chunks) {
#if DORADO_GPU_BUILD && !defined(__APPLE__)
    c10::cuda::OptionalCUDAStreamGuard guard(m_streams[model_id]);
#endif
    return m_caller->call_chunks(model_id, m_input_sigs[model_id], m_input_seqs[model_id],
                                 num_chunks);
}

at::Tensor ModBaseRunner::scale_signal(size_t caller_id,
                                       at::Tensor signal,
                                       const std::vector<int>& seq_ints,
                                       const std::vector<uint64_t>& seq_to_sig_map) const {
    auto& scaler = m_caller->m_caller_data[caller_id]->scaler;
    if (scaler) {
        return scaler->scale_signal(signal, seq_ints, seq_to_sig_map);
    }
    return signal;
}

std::vector<size_t> ModBaseRunner::get_motif_hits(size_t caller_id, const std::string& seq) const {
    return m_caller->m_caller_data[caller_id]->get_motif_hits(seq);
}

const ModBaseModelConfig& ModBaseRunner::caller_params(size_t caller_id) const {
    return m_caller->m_caller_data[caller_id]->params;
}

size_t ModBaseRunner::num_callers() const { return m_caller->m_caller_data.size(); }
void ModBaseRunner::terminate() { m_caller->terminate(); }
void ModBaseRunner::restart() { m_caller->restart(); }

std::string ModBaseRunner::get_name() const {
    std::ostringstream name_stream;
    name_stream << "ModBaseRunner_" << this;
    return name_stream.str();
}

stats::NamedStats ModBaseRunner::sample_stats() const {
    // We don't have direct access to the caller object when the pipeline is set up,
    // so pass through stats here.
    // Each runner will retrieve stats from the caller.
    // Only the last retrieved version will appear, but they should be very similar.
    stats::NamedStats stats = stats::from_obj(*m_caller);
    stats["batches_called"] = double(m_num_batches_called);
    return stats;
}

}  // namespace dorado::modbase
