#include "CudaCRFModel.h"

#include "CRFModelConfig.h"
#include "decode/Decoder.h"
#include "utils/cuda_utils.h"
#include "utils/math_utils.h"

#include <ATen/cuda/CUDAContext.h>
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

namespace dorado::basecall {

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
              m_decoder(decode::create_decoder(device, m_config)),
              m_options(at::TensorOptions().dtype(m_decoder->dtype()).device(device)),
              m_exclusive_gpu_access(exclusive_gpu_access) {
        assert(m_options.device().is_cuda());

        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;
        m_num_input_features = model_config.num_features;
        // adjust chunk size to be a multiple of the stride
        m_out_chunk_size = chunk_size / model_config.stride;
        m_in_chunk_size = m_out_chunk_size * model_config.stride;

        at::InferenceMode guard;
        m_module = load_crf_model(model_config, m_options);

        // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
        // user choice. This makes sure batch size is compatible with GPU kernels.
        if (batch_size == 0) {
            m_batch_size =
                    determine_batch_size(model_config, chunk_size, memory_limit_fraction, true);
        } else {
            int batch_size_granularity = get_batch_size_granularity();
            m_batch_size = utils::pad_to(batch_size, batch_size_granularity);
            // Make sure the requested batch size doesn't exceed the maximum for the memory available.
            auto max_batch_size =
                    determine_batch_size(model_config, chunk_size, memory_limit_fraction, false);
            if (m_batch_size > max_batch_size) {
                spdlog::warn(
                        "Specified batch size {} exceeds maximum batch size based on available "
                        "memory. Using maximum safe batch size {}.",
                        m_batch_size, max_batch_size);
                m_batch_size = max_batch_size;
            }
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

    static int get_batch_size_granularity() {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
    }

    int determine_batch_size(const CRFModelConfig &model_config,
                             int chunk_size_in,
                             float memory_limit_fraction,
                             bool run_benchmark) {
        c10::cuda::CUDAGuard device_guard(m_options.device());
        constexpr float GB = 1.0e9f;
        int64_t available = utils::available_memory(m_options.device());
        spdlog::debug("{} memory available: {:.2f}GB", m_device, available / GB);

#ifdef DORADO_TX2
        return 256;
#endif

        const int granularity = get_batch_size_granularity();

        // If running on a Jetson device with unified memory for CPU and GPU we can't use all
        // the available memory for GPU tasks. This way we leave at least half for the CPU,
        // though it's not clear what the ideal split would be.
        cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
        bool is_unified_memory_device = (prop->major == 5 && prop->minor == 3) ||  // TX1
                                        (prop->major == 6 && prop->minor == 2) ||  // TX2
                                        (prop->major == 7 && prop->minor == 2) ||  // Xavier
                                        (prop->major == 8 && prop->minor == 7);    // Orin
        memory_limit_fraction *= is_unified_memory_device ? 0.5f : 1.f;

        // Apply limit fraction, and allow 1GB for model weights, etc.
        int64_t gpu_mem_limit = int64_t(available * memory_limit_fraction - GB);
        if (gpu_mem_limit < 0) {
            spdlog::warn("Auto batchsize detection failed. Less than 1GB GPU memory available.");
            return granularity;
        }
        spdlog::debug("Auto batchsize {}: memory limit {:.2f}GB", m_device, gpu_mem_limit / GB);

        // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
        // These values have been determined by running dorado with different models and
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
            crfmodel_bytes_per_chunk_timestep = insize_map[model_config.lstm_size];
            if (crfmodel_bytes_per_chunk_timestep == 0) {
                spdlog::warn("Auto batchsize detection failed. Unexpected model insize {}.",
                             model_config.lstm_size);
                return granularity;
            }
        }

        // Determine size of working memory for decoder divided by (batch_size * chunk_size)
        // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
        // where num_states = 4^(state_len+1)
        // See `dorado::basecall::decode::CUDADecoder::beam_search_part_1()` for more details.
        int64_t decode_bytes_per_chunk_timestep =
                10 + m_decoder_options.beam_width * 4 + (1ull << (model_config.state_len * 2 + 2));

        auto bytes_per_chunk_timestep =
                decode_bytes_per_chunk_timestep + crfmodel_bytes_per_chunk_timestep;
        int64_t chunk_size_out = chunk_size_in / model_config.stride;

        int max_batch_size = int(gpu_mem_limit / (bytes_per_chunk_timestep * chunk_size_out));
        max_batch_size -= max_batch_size % granularity;
        if (max_batch_size <= granularity) {
            spdlog::warn("Maximum safe estimated batch size is only {}.", max_batch_size);
            return granularity;
        }

        int best_batch_size = granularity;
        float best_time = std::numeric_limits<float>::max();
        const int chunk_size = std::min(chunk_size_in, model_config.stride * 300);
        if (run_benchmark) {
            // We limit the maximum when doing benchmarking to avoid excessive startup time.
            const int max_batch_size_limit = 10240;
            max_batch_size = std::min(max_batch_size, max_batch_size_limit);
            spdlog::debug("Auto batchsize {}: testing up to {} in steps of {}", m_device,
                          max_batch_size, granularity);
            for (int batch_size = granularity; batch_size <= max_batch_size;
                 batch_size += granularity) {
                auto input = torch::empty({batch_size, model_config.num_features, chunk_size},
                                          m_options);

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

                spdlog::debug("Auto batchsize {}: {}, time per chunk {:8f} ms", m_device,
                              batch_size, time);
                if (time < best_time) {
                    best_time = time;
                    best_batch_size = batch_size;
                }
            }
        } else {
            spdlog::debug("Maximum safe estimated batch size for {}: {}", m_device, max_batch_size);
            best_batch_size = max_batch_size;
        }

        spdlog::debug("Device {} Model memory {:.2f}GB", m_device,
                      (crfmodel_bytes_per_chunk_timestep * chunk_size_out * best_batch_size) / GB);
        spdlog::debug("Device {} Decode memory {:.2f}GB", m_device,
                      (decode_bytes_per_chunk_timestep * chunk_size_out * best_batch_size) / GB);
        return best_batch_size;
    }

    struct NNTask {
        NNTask(at::Tensor input_, int num_chunks_) : input(input_), num_chunks(num_chunks_) {}
        at::Tensor input;
        int num_chunks;
        decode::DecodeData out;
        std::mutex mut;
        std::condition_variable cv;
        bool done{false};
    };

    std::vector<decode::DecodedChunk> call_chunks(at::Tensor &input,
                                                  at::Tensor &output,
                                                  int num_chunks,
                                                  c10::cuda::CUDAStream stream) {
        NVTX3_FUNC_RANGE();
        c10::cuda::CUDAStreamGuard stream_guard(stream);

        if (num_chunks == 0) {
            return std::vector<decode::DecodedChunk>();
        }

        auto task = std::make_shared<NNTask>(input.to(m_options.device()), num_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task->mut);
        while (!task->done) {
            task->cv.wait(lock);
        }

        output.copy_(task->out.data);
        return m_decoder->beam_search_part_2({output, num_chunks, m_decoder_options});
    }

    void cuda_thread_fn() {
        at::InferenceMode guard;
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
#if TORCH_VERSION_MAJOR >= 2
                    print_stat(device_stats.requested_bytes),
#else
                    "unknown",
#endif  // TORCH_VERSION_MAJOR > 1
                    device_stats.num_alloc_retries, device_stats.num_alloc_retries,
                    device_stats.num_ooms, device_stats.max_split_size);

            auto run_basecalling = [&]() {
                stats::Timer timer;
                auto scores = m_module->forward(task->input);
                const auto forward_ms = timer.GetElapsedMS();
                task->out = m_decoder->beam_search_part_1(
                        {scores, task->num_chunks, m_decoder_options});
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
        stats["batches_called"] = double(m_num_batches_called);
        stats["model_ms"] = double(m_model_ms);
        stats["decode_ms"] = double(m_decode_ms);
        return stats;
    }

    const CRFModelConfig m_config;
    std::string m_device;
    std::unique_ptr<decode::Decoder> m_decoder;
    decode::DecoderOptions m_decoder_options;
    at::TensorOptions m_options;
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
    auto opts = at::TensorOptions().device(torch::kCPU).pinned_memory(true);
    m_input = torch::empty(
            {caller->m_batch_size, caller->m_num_input_features, caller->m_in_chunk_size},
            opts.dtype(m_caller->m_options.dtype()));

    m_output = torch::empty({3, caller->m_batch_size, caller->m_out_chunk_size},
                            opts.dtype(torch::kInt8));
}

void CudaModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
    m_input.index_put_({chunk_idx, torch::indexing::Ellipsis}, chunk);
}

std::vector<decode::DecodedChunk> CudaModelRunner::call_chunks(int num_chunks) {
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
    stats["batches_called"] = double(m_num_batches_called);
    return stats;
}

}  // namespace dorado::basecall
