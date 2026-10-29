#pragma once

#include "basecall/DecodedChunk.h"
#include "basecall/ModelCaller.h"
#include "config/BasecallModelConfig.h"

#include <ATen/core/TensorBody.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dorado::basecall {

class MetalModel;
struct MetalCallerTask;
class TensorRunnerState;

class MetalCaller final : public ModelCaller {
public:
    MetalCaller(const config::BasecallModelConfig &model_config, std::unique_ptr<MetalModel> model);
    ~MetalCaller() override;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(TensorRunnerState &state, int num_chunks);
    size_t num_batch_dims() const override { return 1; }

    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks);

    void terminate() override;
    void restart() override;

    const config::BasecallModelConfig &config() const override { return m_config; }
    std::shared_ptr<ModBaseInfo> modbase_info() const override;
    std::string get_name() const override { return "MetalCaller"; }
    stats::NamedStats sample_stats() const override { return {}; }

private:
    MetalCaller(const MetalCaller &) = delete;
    MetalCaller &operator=(const MetalCaller &) = delete;
    MetalCaller(MetalCaller &&) = delete;
    MetalCaller &operator=(MetalCaller &&) = delete;

    void start_threads();
    void metal_thread_fn();
    void decode_thread_fn();

    const config::BasecallModelConfig m_config;
    std::unique_ptr<MetalModel> m_model;

    std::atomic<bool> m_terminate{false};
    std::atomic<bool> m_terminate_decode{false};

    std::deque<std::shared_ptr<MetalCallerTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::thread m_metal_thread;

    std::deque<std::shared_ptr<MetalCallerTask>> m_decode_queue;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::thread> m_decode_threads;
};

}  // namespace dorado::basecall
