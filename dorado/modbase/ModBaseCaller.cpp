#include "ModBaseCaller.h"

#include "ModbaseScaler.h"
#include "MotifMatcher.h"
#include "nn/ModBaseModel.h"
#include "utils/sequence_utils.h"

#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#endif
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <chrono>

using namespace std::chrono_literals;

namespace dorado::modbase {

struct ModBaseCaller::ModBaseTask {
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

ModBaseCaller::ModBaseData::ModBaseData(const std::filesystem::path& model_path,
                                        at::TensorOptions opts,
                                        int batch_size_)
        : module_holder(load_modbase_model(model_path, opts)),
          params(load_modbase_model_config(model_path)),
          matcher(params),
          batch_size(batch_size_) {
    if (params.refine_do_rough_rescale) {
        scaler = std::make_unique<ModBaseScaler>(params.refine_kmer_levels, params.refine_kmer_len,
                                                 params.refine_kmer_center_idx);
    }

#if DORADO_GPU_BUILD && !defined(__APPLE__)
    if (opts.device().is_cuda()) {
        auto sig_len = static_cast<int64_t>(params.context_before + params.context_after);
        auto kmer_len = params.bases_after + params.bases_before + 1;

        // Warmup
        auto input_sigs = torch::empty({batch_size, 1, sig_len}, opts);
        auto input_seqs =
                torch::empty({batch_size, sig_len, utils::BaseInfo::NUM_BASES * kmer_len}, opts);
        module_holder->forward(input_sigs, input_seqs);
        torch::cuda::synchronize(opts.device().index());
    }
#endif
}

std::vector<size_t> ModBaseCaller::ModBaseData::get_motif_hits(const std::string& seq) const {
    return matcher.get_motif_hits(seq);
}

ModBaseCaller::ModBaseCaller(const std::vector<std::filesystem::path>& model_paths,
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
        spdlog::debug("- no metal backend available for modified basecalling, defaulting to CPU.");
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

ModBaseCaller::~ModBaseCaller() {
    m_terminate.store(true);
    for (auto& caller_data : m_caller_data) {
        caller_data->input_cv.notify_one();
    }

    for (auto& task_thread : m_task_threads) {
        task_thread->join();
    }
}

at::Tensor ModBaseCaller::call_chunks(size_t model_id,
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

void ModBaseCaller::terminate() {
    m_terminate.store(true);
    for (auto& caller_data : m_caller_data) {
        caller_data->input_cv.notify_one();
    }
    for (auto& task_thread : m_task_threads) {
        task_thread->join();
    }
    m_task_threads.clear();
}

void ModBaseCaller::restart() {
    if (m_terminate.load()) {
        m_terminate.store(false);
        start_threads();
    }
}

stats::NamedStats ModBaseCaller::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = double(m_num_batches_called);
#if DORADO_GPU_BUILD && !defined(__APPLE__)
    stats["model_ms"] = double(m_model_ms);
#endif
    return stats;
}

void ModBaseCaller::start_threads() {
    for (size_t model_id = 0; model_id < m_num_models; ++model_id) {
        m_task_threads.push_back(std::make_unique<std::thread>(
                &ModBaseCaller::modbase_task_thread_fn, this, model_id));
    }
}

void ModBaseCaller::modbase_task_thread_fn(size_t model_id) {
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

}  // namespace dorado::modbase
