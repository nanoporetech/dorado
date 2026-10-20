#include "CudaCaller.h"

#include "TypedRunnerContext.h"
#include "utils/thread_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <nvtx3/nvtx3.hpp>
#include <torch/cuda.h>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace dorado::basecall {

namespace {

struct NNTask {
    NNTask(CudaModel::Batch batch_, void *caller_) : batch(std::move(batch_)), caller(caller_) {}
    CudaModel::Batch batch;
    void *caller;
    std::mutex mut;
    std::condition_variable cv;
    bool done{false};
};

}  // namespace

// If 5 minutes has passed since the first chunk was added to a batch, we will
// dispatch the batch even if it is not full. This is to prevent issues with
// MinKNOW clients disconnecting because they think the server has timed out.
static constexpr int DEFAULT_FIRST_CHUNK_TIMEOUT_MS = 300000;

// If 30 seconds has passed since the most recent chunk was added to a batch,
// we will dispatch the batch even if it is not full. Benchmarking indicates
// that this gives good results when some pipelines have very low throughput
// and others have very high throughput.
static constexpr int DEFAULT_LAST_CHUNK_TIMEOUT_MS = 30000;

// Default value for timeout of incomplete batches for low-latency pipelines. The
// value of 350 ms has been found to give good adaptive-sampling performance on all
// platforms. For low-latency pipelines the timeout is always from when the first
// chunk was added to the batch.
static constexpr int DEFAULT_LOW_LATENCY_TIMEOUT_MS = 350;

struct CudaCaller::GPUTaskQueue {
public:
    std::queue<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
};

CudaCaller::CudaCaller(const BasecallerCreationParams &params)
        : m_model(std::make_unique<CudaModel>(params)) {
    start_threads();
}

CudaCaller::~CudaCaller() { terminate(); }

std::unique_ptr<ModelCaller::RunnerContext> CudaCaller::create_runner_context(
        size_t batch_dims_idx) {
    return std::make_unique<TypedRunnerContext<CudaCaller, CudaRunnerState>>(*this, *m_model,
                                                                             batch_dims_idx);
}

std::vector<decode::DecodedChunk> CudaCaller::call_chunks(CudaRunnerState &context,
                                                          int num_chunks) {
    c10::cuda::CUDAStreamGuard guard(context.m_stream);
    auto aux = context.create_auxiliary_data();
    auto decoded_chunks = call_chunks(context.m_input, context.m_output, num_chunks, aux.get());
    context.reset();
    return decoded_chunks;
}

CudaCaller::GPUTaskQueue &CudaCaller::get_task_queue() {
    // Global task queues, one per GPU. This ensures that tasks from different clients are
    // processed in the order they became ready, rather than individual callers contending
    // for a global mutex that can wake up threads in an arbitrary order

    static std::vector<GPUTaskQueue> gpu_task_queues(torch::cuda::device_count());
    static std::vector<GPUTaskQueue> low_latency_gpu_task_queues(torch::cuda::device_count());

    auto &task_queues = m_model->is_low_latency() ? low_latency_gpu_task_queues : gpu_task_queues;
    return task_queues.at(m_model->device().index());
}

std::pair<int, int> CudaCaller::batch_timeouts_ms() const {
    // For low-latency pipelines we set both timeouts to the same value. This means that we
    // will always timeout based on the time from the first chunk being added to the batch.
    return m_model->is_low_latency()
                   ? std::make_pair(DEFAULT_LOW_LATENCY_TIMEOUT_MS, DEFAULT_LOW_LATENCY_TIMEOUT_MS)
                   : std::make_pair(DEFAULT_FIRST_CHUNK_TIMEOUT_MS, DEFAULT_LAST_CHUNK_TIMEOUT_MS);
}

std::vector<decode::DecodedChunk> CudaCaller::call_chunks(at::Tensor &input,
                                                          at::Tensor &output,
                                                          int num_chunks,
                                                          nn::AuxiliaryData *const aux) {
    NVTX3_FUNC_RANGE();
    if (num_chunks == 0) {
        return std::vector<decode::DecodedChunk>();
    }

    auto &task_queue = get_task_queue();
    auto task = std::make_shared<NNTask>(m_model->prepare_batch(input, num_chunks, aux), this);
    {
        std::lock_guard<std::mutex> lock(task_queue.m_input_lock);
        task_queue.m_input_queue.push(task);
    }
    task_queue.m_input_cv.notify_all();

    std::unique_lock lock(task->mut);
    while (!task->done) {
        task->cv.wait(lock);
    }

    return m_model->decode_batch(task->batch, output);
}

void CudaCaller::terminate() {
    auto &task_queue = get_task_queue();
    {
        std::lock_guard<std::mutex> lock(task_queue.m_input_lock);
        m_terminate = true;
    }
    task_queue.m_input_cv.notify_all();
    if (m_cuda_thread.joinable()) {
        m_cuda_thread.join();
    }
}

void CudaCaller::restart() {
    // This can be called more than once, via multiple runners.
    auto &task_queue = get_task_queue();
    std::lock_guard<std::mutex> lock(task_queue.m_input_lock);
    if (std::exchange(m_terminate, false)) {
        start_threads();
    }
}

void CudaCaller::start_threads() {
    m_cuda_thread = std::thread([this] { cuda_thread_fn(); });
}

void CudaCaller::cuda_thread_fn() {
    utils::set_thread_name("cuda_caller");
    at::InferenceMode guard;
    const std::string loop_scope_str =
            "cuda_thread_fn_device_" + std::to_string(m_model->device().index());
    const std::string input_q_cv_scope_str =
            "input_queue_cv_device_" + std::to_string(m_model->device().index());

    c10::cuda::CUDAStreamGuard stream_guard(m_model->stream());
    auto &task_queue = get_task_queue();
    while (true) {
        nvtx3::scoped_range loop{loop_scope_str};
        std::unique_lock<std::mutex> input_lock(task_queue.m_input_lock);
        nvtxRangePushA(input_q_cv_scope_str.c_str());
        task_queue.m_input_cv.wait(input_lock, [&] {
            return (!task_queue.m_input_queue.empty() &&
                    task_queue.m_input_queue.front()->caller == this) ||
                   (task_queue.m_input_queue.empty() && m_terminate);
        });
        nvtxRangePop();

        if (task_queue.m_input_queue.empty() && m_terminate) {
            return;
        }

        auto task = task_queue.m_input_queue.front();

        // pop and notify if this is a low latency queue so any other
        // low latency callers that are ready can get going immediately
        if (m_model->is_low_latency()) {
            task_queue.m_input_queue.pop();
            input_lock.unlock();
            task_queue.m_input_cv.notify_all();
        }

        std::unique_lock<std::mutex> task_lock(task->mut);
        m_model->run_batch(task->batch);
        task->done = true;
        task_lock.unlock();
        task->cv.notify_one();

        if (!m_model->is_low_latency()) {
            // not low latency, so pop task and notify callers that we're ready to process a new one.
            // this prevents other callers that use the same GPU from attempting to call a new task
            // before this one has completed, without requiring an unsignalled mutex on the GPU
            // which was shown to have issues waking up threads in an even-handed manner
            task_queue.m_input_queue.pop();
            input_lock.unlock();
            task_queue.m_input_cv.notify_all();
        }
    }
}

}  // namespace dorado::basecall
