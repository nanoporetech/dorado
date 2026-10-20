#include "MetalCallerModel.h"

#include "MetalCallerTask.h"
#include "MetalModel.h"
#include "TensorRunnerState.h"
#include "torch_utils/metal_utils.h"
#include "utils/log_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;

namespace dorado::basecall {

namespace {

class MetalThreadScope final : public ThreadScope {
private:
    utils::ScopedAutoReleasePool m_pool;
};

}  // namespace

MetalCallerModel::MetalCallerModel(const config::BasecallModelConfig &model_config,
                                   std::unique_ptr<MetalModel> model)
        : m_config(model_config), m_model(std::move(model)) {}

MetalCallerModel::~MetalCallerModel() = default;

std::unique_ptr<RunnerState> MetalCallerModel::create_runner_state(size_t batch_dims_idx) {
    if (batch_dims_idx != 0) {
        throw std::out_of_range("Metal caller only has one batch dimension");
    }
    const auto layout = m_config.is_lstm_model() ? TensorLayout::NTC : TensorLayout::NCT;
    return std::make_unique<TensorRunnerState>(m_model->create_input_tensor(), layout);
}

SchedulingPolicy MetalCallerModel::scheduling_policy() const {
    SchedulingPolicy policy;
    policy.release_point = ReleasePoint::on_submit_start;
    policy.decode_threads =
            static_cast<size_t>(std::max(1, utils::get_apple_cpu_perf_core_count() - 1));
    return policy;
}

std::unique_ptr<ThreadScope> MetalCallerModel::worker_thread_scope() {
    // Event IDs restart from 1 whenever the worker thread does.
    m_next_decode_complete_event_id = 1;
    return std::make_unique<MetalThreadScope>();
}

std::unique_ptr<InFlightBatch> MetalCallerModel::prepare(RunnerState &state, int num_chunks) {
    // Input can be NTC or NCT for LSTM and Tx models respectively
    return std::make_unique<MetalCallerTask>(&static_cast<TensorRunnerState &>(state).input(),
                                             num_chunks);
}

void MetalCallerModel::submit(InFlightBatch &batch) {
    auto &task = static_cast<MetalCallerTask &>(batch);
    utils::ScopedAutoReleasePool inner_pool;

    // For unknown reasons, concurrent access to the GPU from multiple instances of this thread --
    // i.e. with > 1 instance of MetalCaller -- results in errors, usually command buffer error code 1.
    // Holding this mutex while executing models seemingly prevents these errors.
    static std::mutex inter_caller_mutex;

    // Assign this task a unique decode completion event ID.
    // This ID will be signalled by the CPU once it has finished relevant decoding work,
    // allowing the GPU to proceed.
    task.decode_complete_event_id = m_next_decode_complete_event_id++;

    // Basecall the chunk and run the scan kernels on GPU
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
        cb_success = m_model->call_task(task, inter_caller_mutex, try_count);
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

size_t MetalCallerModel::decode_units(const InFlightBatch &batch) const {
    // Each chunk is decoded independently.
    return static_cast<size_t>(static_cast<const MetalCallerTask &>(batch).num_chunks);
}

void MetalCallerModel::decode(InFlightBatch & /*batch*/,
                              size_t unit,
                              std::vector<decode::DecodedChunk> &out) {
    auto [sequence, qstring, moves] = m_model->decode(static_cast<int>(unit));
    out[unit] =
            decode::DecodedChunk{std::move(sequence), std::move(qstring), std::move(moves), {}, {}};
}

std::shared_ptr<ModBaseInfo> MetalCallerModel::modbase_info() const {
    return m_model->modbase_info();
}

void MetalCallerModel::release(InFlightBatch &batch, std::vector<decode::DecodedChunk> &out) {
    auto &task = static_cast<MetalCallerTask &>(batch);
    if (auto *decode_complete_event = m_model->decode_complete_event()) {
        // Now that all chunks are decoded, signal that the GPU can overwrite the scores
        // buffer with subsequent work.
        decode_complete_event->setSignaledValue(task.decode_complete_event_id);
    }
    // The sequences are now known, so the modbase models can run.
    m_model->decode_modbase(task, std::span<decode::DecodedChunk>(out.data(), task.num_chunks));
}

}  // namespace dorado::basecall
