#include "ModBaseCaller.h"

#include "ModbaseScaler.h"
#include "MotifMatcher.h"
#include "config/ModBaseModelConfig.h"
#include "nn/ModBaseModel.h"
#include "utils/sequence_utils.h"
#include "utils/thread_naming.h"

#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/cuda.h>
#endif

using namespace std::chrono_literals;

namespace dorado::modbase {

struct ModBaseCaller::ModBaseTask {
    ModBaseTask(at::Tensor input_sigs_, at::Tensor input_seqs_, int num_chunks_)
            : input_sigs(std::move(input_sigs_)),
              input_seqs(std::move(input_seqs_)),
              num_chunks(num_chunks_) {}
    at::Tensor input_sigs;  // Shape: NCT
    at::Tensor input_seqs;  // Shape: NTC (permuted to NCT in model)
    std::mutex mut;
    std::condition_variable cv;
    at::Tensor out;
    bool done{false};
    int num_chunks;
};

ModBaseCaller::ModBaseData::ModBaseData(const config::ModBaseModelConfig& config,
                                        const at::TensorOptions& opts,
                                        const int batch_size_)
        : params(config),
          kmer_refinement_levels(load_kmer_refinement_levels(config)),
          module_holder(load_modbase_model(params, opts)),
          matcher(params),
          batch_size(batch_size_) {
    if (params.refine.do_rough_rescale) {
        scaler = std::make_unique<ModBaseScaler>(kmer_refinement_levels, params.general.kmer_len,
                                                 params.refine.center_idx);
    }

#if DORADO_CUDA_BUILD
    if (opts.device().is_cuda()) {
        c10::cuda::CUDAGuard device_guard(opts.device());
        stream = c10::cuda::getStreamFromPool(false, opts.device().index());

        const int channels = utils::BaseInfo::NUM_BASES * params.general.kmer_len;

        // Warmup
        c10::cuda::CUDAStreamGuard guard(*stream);
        auto input_sigs = torch::empty({batch_size, 1, get_sig_len()}, opts);
        auto input_seqs = torch::empty({batch_size, get_seq_len(), channels}, opts);
        module_holder.forward(input_sigs, input_seqs);
        stream->synchronize();
    }
#endif
}

std::vector<size_t> ModBaseCaller::ModBaseData::get_motif_hits(const std::string& seq) const {
    return matcher.get_motif_hits(seq);
}

int64_t ModBaseCaller::ModBaseData::get_sig_len() const {
    // Depending on the model type, the signal/encoded sequence length either directly
    // defined in the model config, or determined by a chunk size a la canonical base calling.

    const int64_t cs =
            params.is_chunked_input_model() ? params.context.chunk_size : params.context.samples;
    if (cs < 0) {
        throw std::runtime_error("Integer conversion error in ModBaseData::get_sig_len value: '" +
                                 std::to_string(cs) + "'.");
    }
    return cs;
}

int64_t ModBaseCaller::ModBaseData::get_seq_len() const {
    // Depending on the model type, the signal/encoded sequence length either directly
    // defined in the model config, or determined by a chunk size a la canonical base calling.
    return get_sig_len();
}

ModBaseCaller::ModBaseCaller(const std::vector<std::filesystem::path>& model_paths,
                             const int batch_size,
                             const std::string& device)
        : m_num_models(model_paths.size()) {
    if (m_num_models == 0) {
        throw std::logic_error("ModBaseCaller given zero models.");
    }

    if (device == "cpu") {
        // no slow_conv2d_cpu for type Half, need to use float32
        m_options = at::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
#ifdef __APPLE__
    } else if (device == "metal") {
        const auto device_type = torch::kMPS;
        const auto scalar_type = torch::kFloat16;
        m_options = at::TensorOptions().device(device_type).dtype(scalar_type);
#endif
    } else {
        m_options = at::TensorOptions().device(device).dtype(torch::kFloat16);
    }

    // Allocate enough elements up-front so that m_caller_data.emplace_back() doesn't reallocate while
    // other threads can be referencing elements that it's holding.
    m_model_data.reserve(m_num_models);
    m_task_threads.reserve(m_num_models);

    for (size_t i = 0; i < m_num_models; ++i) {
        const auto& config = config::load_modbase_model_config(model_paths[i]);
        at::InferenceMode guard;
        auto caller_data = std::make_unique<ModBaseData>(config, m_options, batch_size);
        m_model_data.emplace_back(std::move(caller_data));
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
    input_sigs.reserve(m_model_data.size());
    for (const auto& caller_data : m_model_data) {
        input_sigs.emplace_back(
                torch::empty({caller_data->batch_size, 1, caller_data->get_sig_len()}, opts));
    }
    return input_sigs;
}

std::vector<at::Tensor> ModBaseCaller::create_input_seq_tensors() const {
    auto opts = at::TensorOptions()
                        .device(torch::kCPU)
                        .pinned_memory(m_options.device().is_cuda())
                        .dtype(torch::kInt8);
    std::vector<at::Tensor> input_seqs;
    input_seqs.reserve(m_model_data.size());
    for (const auto& model_data : m_model_data) {
        const int channels = model_data->params.context.kmer_len * utils::BaseInfo::NUM_BASES;
        const auto seq_len = model_data->get_seq_len();
        input_seqs.emplace_back(torch::empty({model_data->batch_size, seq_len, channels}, opts));
    }
    return input_seqs;
}

at::Tensor ModBaseCaller::call_chunks(size_t model_id,
                                      at::Tensor& input_sigs,
                                      at::Tensor& input_seqs,
                                      int num_chunks) {
    NVTX3_FUNC_RANGE();
    auto& model_data = m_model_data.at(model_id);
    auto task = std::make_shared<ModBaseTask>(input_sigs.to(m_options.device()),
                                              input_seqs.to(m_options.device()), num_chunks);
    {
        std::lock_guard<std::mutex> lock(model_data->input_lock);
        model_data->input_queue.push_front(task);
    }
    model_data->input_cv.notify_one();

    std::unique_lock lock(task->mut);
    while (!task->done) {
        task->cv.wait(lock);
    }

    return task->out.to(torch::kCPU);
}

void ModBaseCaller::terminate() {
    m_terminate.store(true);
    for (auto& model_data : m_model_data) {
        model_data->input_cv.notify_one();
    }
    for (auto& task_thread : m_task_threads) {
        task_thread.join();
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
        m_task_threads.emplace_back([=] { modbase_task_thread_fn(model_id); });
    }
}

void ModBaseCaller::modbase_task_thread_fn(size_t model_id) {
    utils::set_thread_name("modbase_thread");
    auto& model_data = m_model_data[model_id];
#if DORADO_CUDA_BUILD
    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());
    c10::cuda::OptionalCUDAStreamGuard stream_guard(model_data->stream);
#endif
    while (true) {
        nvtx3::scoped_range loop{"modbase_task_thread_fn"};
        at::InferenceMode guard;

        std::unique_lock<std::mutex> input_lock(model_data->input_lock);
        while (model_data->input_queue.empty() && !m_terminate.load()) {
            model_data->input_cv.wait_for(input_lock, 100ms);
        }

        if (model_data->input_queue.empty() && m_terminate.load()) {
            return;
        }

        auto task = model_data->input_queue.back();
        model_data->input_queue.pop_back();
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
        task->out = model_data->module_holder.forward(task->input_sigs, task->input_seqs);
#if DORADO_CUDA_BUILD
        if (model_data->stream.has_value()) {
            model_data->stream->synchronize();
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
