#include "MetalCaller.h"

#include "MetalCallerTask.h"
#include "MetalModel.h"
#include "TensorRunnerState.h"
#include "TypedRunnerContext.h"
#include "torch_utils/metal_utils.h"
#include "utils/log_utils.h"
#include "utils/thread_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>

using namespace dorado::utils;
using namespace std::chrono_literals;

namespace dorado::basecall {

MetalCaller::MetalCaller(const config::BasecallModelConfig &model_config,
                         std::unique_ptr<MetalModel> model)
        : m_config(model_config), m_model(std::move(model)) {
    start_threads();
}

MetalCaller::~MetalCaller() { terminate(); }

std::shared_ptr<ModBaseInfo> MetalCaller::modbase_info() const { return m_model->modbase_info(); }

std::unique_ptr<ModelCaller::RunnerContext> MetalCaller::create_runner_context(
        size_t batch_dims_idx) {
    if (batch_dims_idx != 0) {
        throw std::out_of_range("Metal caller only has one batch dimension");
    }
    const auto layout = m_config.is_lstm_model() ? TensorLayout::NTC : TensorLayout::NCT;
    return std::make_unique<TypedRunnerContext<MetalCaller, TensorRunnerState>>(
            *this, m_model->create_input_tensor(), layout);
}

std::vector<decode::DecodedChunk> MetalCaller::call_chunks(TensorRunnerState &state,
                                                           int num_chunks) {
    std::vector<decode::DecodedChunk> out_chunks(num_chunks);
    call_chunks(state.input(), num_chunks, out_chunks);
    return out_chunks;
}

void MetalCaller::call_chunks(at::Tensor &input,
                              int num_chunks,
                              std::vector<decode::DecodedChunk> &out_chunks) {
    if (num_chunks == 0) {
        return;
    }

    // Input can be NTC or NCT for LSTM and Tx models respectively
    auto task = std::make_shared<MetalCallerTask>(&input, num_chunks, &out_chunks);
    {
        std::lock_guard<std::mutex> lock(m_input_lock);
        m_input_queue.push_front(task);
    }
    m_input_cv.notify_one();

    std::unique_lock lock(task->mut);
    while (!task->done) {
        task->cv.wait(lock);
    }
}

void MetalCaller::terminate() {
    m_terminate.store(true);
    m_input_cv.notify_one();
    m_decode_cv.notify_all();
    if (m_metal_thread.joinable()) {
        m_metal_thread.join();
    }
    for (auto &thr : m_decode_threads) {
        thr.join();
    }
    m_decode_threads.clear();
}

void MetalCaller::restart() {
    // This can be called more than once, via multiple runners.
    if (m_terminate.exchange(false)) {
        m_terminate_decode.store(false);
        start_threads();
    }
}

void MetalCaller::start_threads() {
    m_metal_thread = std::thread([this] { metal_thread_fn(); });

    int num_decode_threads = std::max(1, get_apple_cpu_perf_core_count() - 1);
    m_decode_threads.reserve(num_decode_threads);
    for (int i = 0; i < num_decode_threads; ++i) {
        m_decode_threads.emplace_back([this] { decode_thread_fn(); });
    }
}

void MetalCaller::metal_thread_fn() {
    utils::set_thread_name("metal_worker");
    at::InferenceMode inference_mode_guard;
    ScopedAutoReleasePool outer_pool;

    // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
    // run i before the CPU has finished decoding all run i's chunks.
    // Start at 1, since at event creation ID 0 is deemed to have been signalled.
    auto next_decode_complete_event_id = static_cast<uint64_t>(1);

    // For unknown reasons, concurrent access to the GPU from multiple instances of this thread --
    // i.e. with > 1 instance of MetalCaller -- results in errors, usually command buffer error code 1.
    // Holding this mutex while executing models seemingly prevents these errors.
    static std::mutex inter_caller_mutex;

    while (true) {
        ScopedAutoReleasePool inner_pool;

        // Pop the next task, or return if we're terminated.
        std::shared_ptr<MetalCallerTask> task;
        {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate.load()) {
                m_input_cv.wait_for(input_lock, 100ms);
            }

            if (m_input_queue.empty() && m_terminate.load()) {
                m_terminate_decode.store(true);
                return;
            }

            task = std::move(m_input_queue.back());
            m_input_queue.pop_back();
        }

        // Assign this task a unique decode completion event ID.
        // This ID will be signalled by the CPU once it has finished relevant decoding work,
        // allowing the GPU to proceed.
        task->decode_complete_event_id = next_decode_complete_event_id++;

        // Basecall the chunk and run the scan kernels on GPU
        {
            auto retry_delay = 100ms;
            auto sleep_before_retry = [&] {
                std::this_thread::sleep_for(retry_delay);
                retry_delay *= 2;
                // These are rare enough that sleeping for a few seconds shouldn't impact
                // speed, and should give the system plenty of time to recover.
                if (retry_delay > 5s) {
                    retry_delay = 5s;
                }
            };

            // We retry the entire set of kernels up to 5 times, to deal with seemingly
            // random intermittent errors with command buffer submissions.
            // TODO: find a more robust way of dealing with Metal kernel launch issues
            bool cb_success = false;
            for (int try_count = 0; try_count < 5; ++try_count) {
                cb_success = m_model->call_task(*task, inter_caller_mutex, try_count);
                if (cb_success) {
                    break;
                } else {
                    sleep_before_retry();
                }
            }

            // If we repeatedly submitted CBs without success, we give up.
            if (!cb_success) {
                DORADO_LOG(critical, "Exiting. Failed to successfully submit GPU command buffers.");
                std::exit(EXIT_FAILURE);
            }
        }

        // Pass task on to decode threads
        {
            std::lock_guard decode_lock(m_decode_lock);
            m_decode_queue.push_front(std::move(task));
        }
        m_decode_cv.notify_all();
    }
}

void MetalCaller::decode_thread_fn() {
    utils::set_thread_name("metal_decode");
    at::InferenceMode inference_mode_guard;
    while (true) {
        std::unique_lock<std::mutex> decode_lock(m_decode_lock);
        while (m_decode_queue.empty() && !m_terminate_decode.load()) {
            m_decode_cv.wait_for(decode_lock, 100ms);
        }

        if (m_decode_queue.empty() && m_terminate_decode.load()) {
            return;
        }
        auto task = m_decode_queue.back();
        int chunk_idx = task->decode_chunks_started++;
        // If all chunks have been picked up for decoding, remove task from queue
        if (chunk_idx == task->num_chunks - 1) {
            m_decode_queue.pop_back();
        }
        decode_lock.unlock();

        auto [sequence, qstring, moves] = m_model->decode(chunk_idx);
        (*task->out_chunks)[chunk_idx] = decode::DecodedChunk{
                std::move(sequence), std::move(qstring), std::move(moves), {}, {}};

        // Wake the waiting thread which called `call_chunks()` if we're done decoding
        std::unique_lock<std::mutex> task_lock(task->mut);
        bool all_decoded = ++(task->decode_chunks_finished) == task->num_chunks;
        task_lock.unlock();
        if (all_decoded) {
            if (auto *decode_complete_event = m_model->decode_complete_event()) {
                // Now that all chunks are decoded, signal that the GPU can overwrite the scores
                // buffer with subsequent work.
                decode_complete_event->setSignaledValue(task->decode_complete_event_id);
            }
            // The sequences are now known, so the modbase models can run.
            m_model->decode_modbase(*task);
            {
                std::lock_guard<std::mutex> done_lock(task->mut);
                task->done = true;
            }
            task->cv.notify_one();
        }
    }
}

}  // namespace dorado::basecall
