#include "CudaCRFModel.h"

#include "CRFModelConfig.h"
#include "decode/GPUDecoder.h"
#include "utils/cuda_utils.h"
#include "utils/math_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nvtx3/nvtx3.hpp>
#include <spdlog/spdlog.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <limits>

using namespace std::chrono_literals;

namespace dorado {

class CudaCaller {
public:
    CudaCaller(const CRFModelConfig &model_config,
               int chunk_size,
               int batch_size,
               const std::string &device,
               float memory_limit_fraction,
               bool exclusive_gpu_access)
            : m_config(model_config),
              m_device(device),
              m_exclusive_gpu_access(exclusive_gpu_access) {
        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;
        m_decoder = std::make_unique<GPUDecoder>();
        m_num_input_features = model_config.num_features;
        // adjust chunk size to be a multiple of the stride
        m_out_chunk_size = chunk_size / model_config.stride;
        m_in_chunk_size = m_out_chunk_size * model_config.stride;

        m_options = torch::TensorOptions().dtype(GPUDecoder::dtype).device(device);
        assert(m_options.device().is_cuda());

        torch::InferenceMode guard;
        m_module = load_crf_model(model_config, m_options);

        // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
        // user choice. This makes sure batch size is compatible with GPU kernels.
        if (batch_size == 0) {
            m_batch_size = auto_batch_size(model_config, chunk_size, memory_limit_fraction);
        } else {
            int batch_size_granularity = get_batch_size_granularity(model_config, m_options);
            m_batch_size = utils::pad_to(batch_size, batch_size_granularity);
            // Warmup
            auto input =
                    torch::empty({m_batch_size, m_num_input_features, m_in_chunk_size}, m_options);
            m_module->forward(input);
            torch::cuda::synchronize(m_options.device().index());
        }

        c10::cuda::CUDAGuard device_guard(m_options.device());
        c10::cuda::CUDACachingAllocator::emptyCache();

        start_threads();
    }

    void start_threads() {
        m_cuda_thread.reset(new std::thread(&CudaCaller::cuda_thread_fn, this));
    }

    ~CudaCaller() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        if (m_cuda_thread && m_cuda_thread->joinable()) {
            m_cuda_thread->join();
        }
    }

    static int get_batch_size_granularity(const CRFModelConfig &model_config,
                                          const torch::TensorOptions &options) {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
    }

    int auto_batch_size(const dorado::CRFModelConfig &model_config,
                        int chunk_size_in,
                        float memory_limit_fraction) {
#ifdef DORADO_TX2
        return 256;
#else
        int64_t available = utils::available_memory(m_options.device()) * memory_limit_fraction;
        spdlog::debug("Auto batch size: GPU memory available: {}GB", available / 1.0e9f);

        int granularity = get_batch_size_granularity(model_config, m_options);

        // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
        // These values have been derermined by running dorado with different models and
        // reporting the actual allocation size per chunk-timestep.
        int64_t crfmodel_bytes_per_chunk_timestep;
        if (model_config.out_features.has_value()) {
            auto out_features = model_config.out_features.value();
            std::unordered_map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}};
            crfmodel_bytes_per_chunk_timestep = out_features_map[out_features];
            if (crfmodel_bytes_per_chunk_timestep == 0) {
                spdlog::warn("Auto batchsize detection failed. Unexpected model out_features {}.",
                             out_features);
                return granularity;
            }
        } else {
            std::unordered_map<int, int64_t> insize_map{
                    {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
            crfmodel_bytes_per_chunk_timestep = insize_map[model_config.insize];
            if (crfmodel_bytes_per_chunk_timestep == 0) {
                spdlog::warn("Auto batchsize detection failed. Unexpected model insize {}.",
                             model_config.insize);
                return granularity;
            }
        }

        // Determine size of working memory for decoder divided by (batch_size * chunk_size)
        // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
        // where num_states = 4^(state_len+1)
        // See `dorado::GPUDecoder::gpu_part()`, block beginning with `if (!initialized) {`
        // for more details.
        int64_t decode_bytes_per_chunk_timestep =
                10 + m_decoder_options.beam_width * 4 + (1 << (model_config.state_len * 2 + 2));

        auto bytes_per_chunk_timestep =
                decode_bytes_per_chunk_timestep + crfmodel_bytes_per_chunk_timestep;
        int64_t chunk_size_out = chunk_size_in / model_config.stride;
        available = available - 1.0e9f;  // Allow 1GB for model weights, etc.
        if (available < 0) {
            spdlog::warn("Auto batchsize detection failed. Less than 1GB GPU memory available.");
            return granularity;
        }

        const int64_t max_batch_size_limit = 10240;
        const int max_batch_size = std::min(available / (bytes_per_chunk_timestep * chunk_size_out),
                                            max_batch_size_limit);
        if (max_batch_size < utils::pad_to(128, granularity) + granularity) {
            spdlog::warn("Auto batchsize detection failed. Estimated max batch size only {}.",
                         max_batch_size);
            return granularity;
        }

        c10::cuda::CUDAGuard device_guard(m_options.device());

        int best_batch_size = granularity;
        float best_time = std::numeric_limits<float>::max();
        const int chunk_size = std::min(chunk_size_in, model_config.stride * 300);
        spdlog::debug("Auto batch size: testing up to {} in steps of {}", max_batch_size,
                      granularity);
        for (int batch_size = granularity; batch_size <= max_batch_size;
             batch_size += granularity) {
            auto input =
                    torch::empty({batch_size, model_config.num_features, chunk_size}, m_options);

            float time = std::numeric_limits<float>::max();
            for (int i = 0; i < 2; ++i) {  // run twice to eliminate outliers
                using utils::handle_cuda_result;
                cudaEvent_t start, stop;
                handle_cuda_result(cudaEventCreate(&start));
                handle_cuda_result(cudaEventCreate(&stop));
                handle_cuda_result(cudaEventRecord(start));
                m_module->forward(input);
                handle_cuda_result(cudaEventRecord(stop));
                handle_cuda_result(cudaEventSynchronize(stop));
                float ms = 0;
                handle_cuda_result(cudaEventElapsedTime(&ms, start, stop));
                time = std::min(time, ms / batch_size);
                handle_cuda_result(cudaEventDestroy(start));
                handle_cuda_result(cudaEventDestroy(stop));
            }

            spdlog::debug("Auto batchsize {}: {}, time per chunk {} ms", m_device, batch_size,
                          time);
            if (time < best_time) {
                best_time = time;
                best_batch_size = batch_size;
            }
        }

        spdlog::debug(
                "Device {} Model memory {}", m_device,
                (crfmodel_bytes_per_chunk_timestep * chunk_size_out * best_batch_size) / 1e9f);
        spdlog::debug("Device {} Decode memory {}", m_device,
                      (decode_bytes_per_chunk_timestep * chunk_size_out * best_batch_size) / 1e9f);
        return best_batch_size;
#endif
    }

    struct NNTask {
        NNTask(torch::Tensor input_, torch::Tensor &output_, int num_chunks_)
                : input(input_), out(output_), num_chunks(num_chunks_) {}
        torch::Tensor input;
        torch::Tensor &out;
        std::mutex mut;
        std::condition_variable cv;
        bool done{false};
        int num_chunks;
    };

    std::vector<DecodedChunk> call_chunks(torch::Tensor &input,
                                          torch::Tensor &output,
                                          int num_chunks,
                                          c10::cuda::CUDAStream stream) {
        NVTX3_FUNC_RANGE();
        c10::cuda::CUDAStreamGuard stream_guard(stream);

        if (num_chunks == 0) {
            return std::vector<DecodedChunk>();
        }

        auto task = std::make_shared<NNTask>(input, output, num_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task->mut);
        while (!task->done) {
            task->cv.wait(lock);
        }

        return m_decoder->cpu_part(output);
    }

    void cuda_thread_fn() {
        torch::InferenceMode guard;
        c10::cuda::CUDAGuard device_guard(m_options.device());
        auto stream = c10::cuda::getCurrentCUDAStream(m_options.device().index());

        const std::string loop_scope_str =
                "cuda_thread_fn_device_" + std::to_string(m_options.device().index());
        const std::string input_q_cv_scope_str =
                "input_queue_cv_device_" + std::to_string(m_options.device().index());
        const std::string gpu_lock_scope_str =
                "gpu_lock_" + std::to_string(m_options.device().index());
        while (true) {
            nvtx3::scoped_range loop{loop_scope_str};
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            nvtxRangePushA(input_q_cv_scope_str.c_str());
            while (m_input_queue.empty() && !m_terminate.load()) {
                m_input_cv.wait_for(input_lock, 100ms);
            }
            nvtxRangePop();

            if (m_input_queue.empty() && m_terminate.load()) {
                return;
            }

            auto task = m_input_queue.back();
            m_input_queue.pop_back();
            input_lock.unlock();

            nvtxRangePushA(gpu_lock_scope_str.c_str());
            auto gpu_lock = dorado::utils::acquire_gpu_lock(m_options.device().index(),
                                                            m_exclusive_gpu_access);
            nvtxRangePop();

            std::unique_lock<std::mutex> task_lock(task->mut);

#ifndef DORADO_TX2
            auto device_stats =
                    c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());

            auto print_stat = [](c10::cuda::CUDACachingAllocator::StatArray &st) {
                std::string s("");
                s += "aggregate current " + std::to_string(st[0].current);
                s += "\n";
                return s;
            };
            spdlog::trace(
                    "allocation {}, segment {}, active {}, inactive_split {}, alloc_bytes {}, "
                    "reserved_bytes {}, active_bytes {}, inactive_split_bytes {}, requested_bytes "
                    "{}, num_alloc_retries {}, num_ooms {}, max_split_size {}",
                    print_stat(device_stats.allocation), print_stat(device_stats.segment),
                    print_stat(device_stats.active), print_stat(device_stats.inactive_split),
                    print_stat(device_stats.allocated_bytes),
                    print_stat(device_stats.reserved_bytes), print_stat(device_stats.active_bytes),
                    print_stat(device_stats.inactive_split_bytes),
                    print_stat(device_stats.requested_bytes), device_stats.num_alloc_retries,
                    device_stats.num_alloc_retries, device_stats.num_ooms,
                    device_stats.max_split_size);
#endif  // #ifndef DORADO_TX2
            auto run_basecalling = [&]() {
                stats::Timer timer;
                auto scores = m_module->forward(task->input.to(m_options.device(), true));
                const auto forward_ms = timer.GetElapsedMS();
                task->out.copy_(m_decoder->gpu_part(scores, task->num_chunks, m_decoder_options));
                stream.synchronize();
                const auto forward_plus_decode_ms = timer.GetElapsedMS();
                m_model_ms += forward_ms;
                m_decode_ms += forward_plus_decode_ms - forward_ms;
            };

            try {
                run_basecalling();
            } catch (c10::Error &e) {
                spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
                c10::cuda::CUDACachingAllocator::emptyCache();
                run_basecalling();
            }
            ++m_num_batches_called;
            task->done = true;
            task_lock.unlock();
            task->cv.notify_one();
        }
    }

    void terminate() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        if (m_cuda_thread && m_cuda_thread->joinable()) {
            m_cuda_thread->join();
        }
        m_cuda_thread.reset();
    }

    void restart() {
        // This can be called more than one, via multiple runners.
        if (m_terminate.load()) {
            m_terminate.store(false);
            start_threads();
        }
    }

    std::string get_name() const { return std::string("CudaCaller_") + m_device; }

    stats::NamedStats sample_stats() const {
        stats::NamedStats stats;
        stats["batches_called"] = m_num_batches_called;
        stats["model_ms"] = m_model_ms;
        stats["decode_ms"] = m_decode_ms;
        return stats;
    }

    const CRFModelConfig m_config;
    std::string m_device;
    torch::TensorOptions m_options;
    std::unique_ptr<GPUDecoder> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    std::atomic<bool> m_terminate{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_cuda_thread;
    int m_num_input_features, m_batch_size, m_in_chunk_size, m_out_chunk_size;
    bool m_exclusive_gpu_access;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
    std::atomic<int64_t> m_model_ms = 0;
    std::atomic<int64_t> m_decode_ms = 0;
};

std::shared_ptr<CudaCaller> create_cuda_caller(const CRFModelConfig &model_config,
                                               int chunk_size,
                                               int batch_size,
                                               const std::string &device,
                                               float memory_limit_fraction,
                                               bool exclusive_gpu_access) {
    return std::make_shared<CudaCaller>(model_config, chunk_size, batch_size, device,
                                        memory_limit_fraction, exclusive_gpu_access);
}

CudaModelRunner::CudaModelRunner(std::shared_ptr<CudaCaller> caller)
        : m_caller(caller),
          m_stream(c10::cuda::getStreamFromPool(false, m_caller->m_options.device().index())) {
    auto opts = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    m_input = torch::empty(
            {caller->m_batch_size, caller->m_num_input_features, caller->m_in_chunk_size},
            opts.dtype(m_caller->m_options.dtype()));

    m_output = torch::empty({3, caller->m_batch_size, caller->m_out_chunk_size},
                            opts.dtype(torch::kInt8));
}

void CudaModelRunner::accept_chunk(int chunk_idx, const torch::Tensor &chunk) {
    m_input.index_put_({chunk_idx, torch::indexing::Ellipsis}, chunk);
}

std::vector<DecodedChunk> CudaModelRunner::call_chunks(int num_chunks) {
    ++m_num_batches_called;
    stats::Timer timer;
    auto decoded_chunks = m_caller->call_chunks(m_input, m_output, num_chunks, m_stream);
    return decoded_chunks;
}

const CRFModelConfig &CudaModelRunner::config() const { return m_caller->m_config; }
size_t CudaModelRunner::model_stride() const { return m_caller->m_config.stride; }
size_t CudaModelRunner::chunk_size() const { return m_input.size(2); }
size_t CudaModelRunner::batch_size() const { return m_input.size(0); }
void CudaModelRunner::terminate() { m_caller->terminate(); }
void CudaModelRunner::restart() { m_caller->restart(); }

std::string CudaModelRunner::get_name() const {
    // The name must be unique across multiple instances.
    // We could take a unique ID at setup time, but for now just use the address.
    std::ostringstream name_stream;
    name_stream << "CudaModelRunner_" << this;
    return name_stream.str();
}

stats::NamedStats CudaModelRunner::sample_stats() const {
    // We don't have direct access to the caller object when the pipeline is set up,
    // so pass through stats here.
    // Each runner will retrieve stats from the caller.
    // Only the last retrieved version will appear, but they should be very similar.
    stats::NamedStats stats = stats::from_obj(*m_caller);
    stats["batches_called"] = m_num_batches_called;
    return stats;
}

}  // namespace dorado
