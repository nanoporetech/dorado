#include "CudaCRFModel.h"

#include "decode/GPUDecoder.h"
#include "utils/cuda_utils.h"
#include "utils/math_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <nvtx3/nvtx3.hpp>
#include <toml.hpp>
#include <torch/torch.h>

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
            : m_device(device), m_exclusive_gpu_access(exclusive_gpu_access) {
        m_model_stride = static_cast<size_t>(model_config.stride);

        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;
        m_decoder = std::make_unique<GPUDecoder>();
        m_num_input_features = model_config.num_features;
        // adjust chunk size to be a multiple of the stride
        m_out_chunk_size = chunk_size / m_model_stride;
        m_in_chunk_size = m_out_chunk_size * m_model_stride;

        m_options = torch::TensorOptions().dtype(GPUDecoder::dtype).device(device);
        assert(m_options.device().is_cuda());

        torch::InferenceMode guard;
        m_module = load_crf_model(model_config, m_options);

        // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
        // user choice. This makes sure batch size is compatible with GPU kernels.
        int batch_size_granularity = get_batch_size_granularity(model_config, m_options);
        m_batch_size = utils::pad_to(batch_size, batch_size_granularity);
        if (batch_size == 0) {
            m_batch_size =
                    utils::auto_gpu_batch_size(m_module, model_config, m_options,
                                               batch_size_granularity, memory_limit_fraction);
        } else {
            // Warmup
            auto input =
                    torch::empty({m_batch_size, m_num_input_features, m_in_chunk_size}, m_options);
            m_module->forward(input);
            torch::cuda::synchronize(m_options.device().index());
        }

        m_cuda_thread.reset(new std::thread(&CudaCaller::cuda_thread_fn, this));
    }

    ~CudaCaller() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        m_cuda_thread->join();
    }

    static int get_batch_size_granularity(const CRFModelConfig &model_config,
                                          const torch::TensorOptions &options) {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
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
        auto task = std::make_shared<NNTask>(input.to(m_options.device()), output, num_chunks);
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
            stats::Timer timer;
            auto scores = m_module->forward(task->input);
            const auto forward_ms = timer.GetElapsedMS();
            task->out.copy_(m_decoder->gpu_part(scores, task->num_chunks, m_decoder_options));
            stream.synchronize();
            const auto forward_plus_decode_ms = timer.GetElapsedMS();
            ++m_num_batches_called;
            m_model_ms += forward_ms;
            m_decode_ms += forward_plus_decode_ms - forward_ms;
            task->done = true;
            task_lock.unlock();
            task->cv.notify_one();
        }
    }
    void terminate() { m_terminate.store(true); }

    std::string get_name() const { return std::string("CudaCaller_") + m_device; }

    stats::NamedStats sample_stats() const {
        stats::NamedStats stats;
        stats["batches_called"] = m_num_batches_called;
        stats["model_ms"] = m_model_ms;
        stats["decode_ms"] = m_decode_ms;
        return stats;
    }

    std::string m_device;
    torch::TensorOptions m_options;
    std::unique_ptr<GPUDecoder> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    size_t m_model_stride;
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

size_t CudaModelRunner::model_stride() const { return m_caller->m_model_stride; }
size_t CudaModelRunner::chunk_size() const { return m_input.size(2); }
size_t CudaModelRunner::batch_size() const { return m_input.size(0); }
void CudaModelRunner::terminate() { m_caller->terminate(); }

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
