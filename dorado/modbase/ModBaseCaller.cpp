#include "ModBaseCaller.h"

#include "ModbaseScaler.h"
#include "MotifMatcher.h"
#include "nn/ModBaseModel.h"
#include "utils/sequence_utils.h"

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#endif
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <chrono>

using namespace std::chrono_literals;

namespace dorado::modbase {

struct ModBaseCaller::ModBaseTask {
    ModBaseTask(at::Tensor input_sigs_, at::Tensor input_seqs_, int num_chunks_)
            : input_sigs(std::move(input_sigs_)),
              input_seqs(std::move(input_seqs_)),
              num_chunks(num_chunks_) {}
    at::Tensor input_sigs;
    at::Tensor input_seqs;
    std::mutex mut;
    std::condition_variable cv;
    at::Tensor out;
    bool done{false};
    int num_chunks;
};

ModBaseCaller::ModBaseData::ModBaseData(const std::filesystem::path& model_path,
                                        at::TensorOptions opts,
                                        int batch_size_)
        : params(load_modbase_model_config(model_path)),
          module_holder(load_modbase_model(model_path, opts)),
          matcher(params),
          batch_size(batch_size_) {
    if (params.refine_do_rough_rescale) {
        scaler = std::make_unique<ModBaseScaler>(params.refine_kmer_levels, params.refine_kmer_len,
                                                 params.refine_kmer_center_idx);
    }

#if DORADO_CUDA_BUILD
    if (opts.device().is_cuda()) {
        stream = c10::cuda::getStreamFromPool(false, opts.device().index());

        auto sig_len = static_cast<int64_t>(params.context_before + params.context_after);
        auto kmer_len = params.bases_after + params.bases_before + 1;

        // Warmup
        c10::cuda::OptionalCUDAStreamGuard guard(stream);
        auto input_sigs = torch::empty({batch_size, 1, sig_len}, opts);
        auto input_seqs =
                torch::empty({batch_size, sig_len, utils::BaseInfo::NUM_BASES * kmer_len}, opts);
        module_holder->forward(input_sigs, input_seqs);
        stream->synchronize();
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
#ifdef __APPLE__
    } else if (device == "metal") {
#if TARGET_OS_IPHONE
        spdlog::warn("Using CPU backend since no MPS backend available on iOS.");
        const auto device_type = torch::kCPU;
        const auto scalar_type = torch::kFloat32;
#else
        const auto device_type = torch::kMPS;
        const auto scalar_type = torch::kFloat16;
#endif
        m_options = at::TensorOptions().device(device_type).dtype(scalar_type);
#endif
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

ModBaseCaller::~ModBaseCaller() { terminate(); }

std::vector<at::Tensor> ModBaseCaller::create_input_sig_tensors() const {
    auto opts = at::TensorOptions()
                        .device(torch::kCPU)
                        .pinned_memory(m_options.device().is_cuda())
                        .dtype(m_options.dtype());

    std::vector<at::Tensor> input_sigs;
    for (auto& caller_data : m_caller_data) {
        auto sig_len = static_cast<int64_t>(caller_data->params.context_before +
                                            caller_data->params.context_after);
        input_sigs.push_back(torch::empty({caller_data->batch_size, 1, sig_len}, opts));
    }
    return input_sigs;
}

std::vector<at::Tensor> ModBaseCaller::create_input_seq_tensors() const {
    auto opts = at::TensorOptions()
                        .device(torch::kCPU)
                        .pinned_memory(m_options.device().is_cuda())
                        .dtype(torch::kInt8);

    std::vector<at::Tensor> input_seqs;
    for (auto& caller_data : m_caller_data) {
        auto sig_len = static_cast<int64_t>(caller_data->params.context_before +
                                            caller_data->params.context_after);
        auto kmer_len = caller_data->params.bases_after + caller_data->params.bases_before + 1;
        input_seqs.push_back(torch::empty(
                {caller_data->batch_size, sig_len, utils::BaseInfo::NUM_BASES * kmer_len}, opts));
    }
    return input_seqs;
}

at::Tensor ModBaseCaller::call_chunks(size_t model_id,
                                      at::Tensor& input_sigs,
                                      at::Tensor& input_seqs,
                                      int num_chunks) {
    NVTX3_FUNC_RANGE();
    auto& caller_data = m_caller_data[model_id];
    auto task = std::make_shared<ModBaseTask>(input_sigs.to(m_options.device()),
                                              input_seqs.to(m_options.device()), num_chunks);
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
    if (m_terminate.exchange(false)) {
        start_threads();
    }
}

stats::NamedStats ModBaseCaller::sample_stats() const {
    stats::NamedStats stats;
    stats["batches_called"] = double(m_num_batches_called);
#if DORADO_CUDA_BUILD
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
#if DORADO_CUDA_BUILD
    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());
    c10::cuda::OptionalCUDAStreamGuard stream_guard(caller_data->stream);
#endif
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

#if DORADO_CUDA_BUILD
        auto gpu_lock = [&] {
            if (m_options.device().is_cuda()) {
                return std::unique_lock(gpu_mutexes[m_options.device().index()]);
            }
            return std::unique_lock<std::mutex>{};
        }();
#endif
        std::unique_lock<std::mutex> task_lock(task->mut);
        stats::Timer timer;
        task->out = caller_data->module_holder->forward(task->input_sigs, task->input_seqs);
#if DORADO_CUDA_BUILD
        if (caller_data->stream.has_value()) {
            caller_data->stream->synchronize();
        }
#endif
        // Only meaningful if we're syncing the stream.
        m_model_ms += timer.GetElapsedMS();
        ++m_num_batches_called;
        task->done = true;
        task_lock.unlock();
        task->cv.notify_one();
    }
}

}  // namespace dorado::modbase
