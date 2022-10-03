#include "CudaCRFModel.h"

#include "decode/GPUDecoder.h"

#include <c10/cuda/CUDAGuard.h>
#include <toml.hpp>
#include <torch/torch.h>

using namespace std::chrono_literals;

class CudaCaller {
public:
    CudaCaller(const std::filesystem::path &model_path,
               int chunk_size,
               int batch_size,
               const std::string &device) {
        auto config = toml::parse(model_path / "config.toml");
        const auto &qscore = toml::find(config, "qscore");
        const auto qbias = toml::find<float>(qscore, "bias");
        const auto qscale = toml::find<float>(qscore, "scale");

        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = qbias;
        m_decoder_options.q_scale = qscale;
        m_decoder = std::make_unique<GPUDecoder>();

        m_options = torch::TensorOptions().dtype(GPUDecoder::dtype).device(device);
        auto [crf_module, stride] = load_crf_model(model_path, batch_size, chunk_size, m_options);
        m_module = crf_module;
        m_model_stride = stride;

        m_cuda_thread.reset(new std::thread(&CudaCaller::cuda_thread_fn, this));
    }

    ~CudaCaller() {
        std::unique_lock<std::mutex> input_lock(m_input_lock);
        m_terminate = true;
        input_lock.unlock();
        m_input_cv.notify_one();
        m_cuda_thread->join();
    }

    struct NNTask {
        NNTask(torch::Tensor *input_, int num_chunks_) : input(input_), num_chunks(num_chunks_) {}

        torch::Tensor *input;
        std::mutex mut;
        std::condition_variable cv;
        torch::Tensor out;
        bool done{false};
        int num_chunks;
    };

    std::vector<DecodedChunk> call_chunks(torch::Tensor &input, int num_chunks) {
        if (num_chunks == 0) {
            return std::vector<DecodedChunk>();
        }

        NNTask task(&input, num_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(&task);
        }
        m_input_cv.notify_one();

        std::unique_lock lock(task.mut);
        while (!task.done) {
            task.cv.wait(lock);
        }
        return m_decoder->cpu_part(task.out);
    }

    void cuda_thread_fn() {
        torch::InferenceMode guard;
        c10::cuda::CUDAGuard device_guard(m_options.device_opt().value());

        while (true) {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate) {
                m_input_cv.wait_for(input_lock, 100ms);
            }
            // TODO: finish work before terminating?
            if (m_terminate) {
                return;
            }

            NNTask *task = m_input_queue.back();
            m_input_queue.pop_back();
            input_lock.unlock();

            auto scores = m_module->forward(
                    task->input->to(m_options.device_opt().value()).to(GPUDecoder::dtype));
            std::unique_lock<std::mutex> task_lock(task->mut);
            task->out = m_decoder->gpu_part(scores, task->num_chunks, m_decoder_options);
            task->done = true;
            task_lock.unlock();
            task->cv.notify_one();
        }
    }

    std::string m_device;
    torch::TensorOptions m_options;
    std::unique_ptr<GPUDecoder> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    size_t m_model_stride;
    bool m_terminate{false};
    std::deque<NNTask *> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_cuda_thread;
};

std::shared_ptr<CudaCaller> create_cuda_caller(const std::filesystem::path &model_path,
                                               int chunk_size,
                                               int batch_size,
                                               const std::string &device) {
    return std::make_shared<CudaCaller>(model_path, chunk_size, batch_size, device);
}

CudaModelRunner::CudaModelRunner(std::shared_ptr<CudaCaller> caller, int chunk_size, int batch_size)
        : m_caller(caller) {
    m_input = torch::empty(
            {batch_size, 1, chunk_size},
            torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU).pinned_memory(true));
    // warm up
    call_chunks(batch_size);
}

void CudaModelRunner::accept_chunk(int chunk_idx, at::Tensor slice) {
    m_input.index_put_({chunk_idx, 0}, slice);
}

std::vector<DecodedChunk> CudaModelRunner::call_chunks(int num_chunks) {
    return m_caller->call_chunks(m_input, num_chunks);
}

size_t CudaModelRunner::model_stride() const { return m_caller->m_model_stride; }
