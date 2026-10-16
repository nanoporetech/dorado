#pragma once

#include "DecodedChunk.h"
#include "ModelCaller.h"
#include "config/BasecallModelConfig.h"
#include "torch_utils/metal_utils.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace dorado::basecall {

class TensorRunnerState;

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalCaller : public ModelCaller {
protected:
    explicit MetalCaller(const config::BasecallModelConfig &model_config)
            : m_config(model_config) {}

public:
    static std::shared_ptr<MetalCaller> create(const config::BasecallModelConfig &model_config,
                                               float memory_limit_fraction);
    static int get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &config);
    static int get_batch_size_granularity(const config::BasecallModelConfig &config);

    ~MetalCaller() override;

    std::unique_ptr<RunnerContext> create_runner_context(size_t batch_dims_idx) override;
    std::vector<decode::DecodedChunk> call_chunks(TensorRunnerState &state, int num_chunks);
    size_t num_batch_dims() const override { return 1; }

    virtual at::Tensor create_input_tensor() const = 0;
    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks);

    void terminate() override;
    void restart() override;

    const config::BasecallModelConfig &config() const override { return m_config; }
    std::shared_ptr<ModBaseInfo> modbase_info() const override { return nullptr; }
    std::string get_name() const override { return "MetalCaller"; }
    stats::NamedStats sample_stats() const override { return {}; }

    struct NNTask;

private:
    MetalCaller(const MetalCaller &) = delete;
    MetalCaller &operator=(const MetalCaller &) = delete;
    MetalCaller(MetalCaller &&) = delete;
    MetalCaller &operator=(MetalCaller &&) = delete;

protected:
    void start_threads();
    void metal_thread_fn();
    void decode_thread_fn();

    virtual DecodedData decode(int chunk_idx) const = 0;
    virtual bool call_task(NNTask &task, std::mutex &inter_caller_mutex, int try_count) = 0;

    const config::BasecallModelConfig m_config;

    std::atomic<bool> m_terminate{false};
    std::atomic<bool> m_terminate_decode{false};

    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::thread m_metal_thread;

    std::deque<std::shared_ptr<NNTask>> m_decode_queue;
    std::mutex m_decode_lock;
    std::condition_variable m_decode_cv;
    std::vector<std::thread> m_decode_threads;
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    NS::SharedPtr<MTL::Device> m_device;
};

}  // namespace dorado::basecall
