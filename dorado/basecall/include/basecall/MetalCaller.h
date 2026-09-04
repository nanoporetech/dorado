#pragma once

#include "DecodedChunk.h"
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

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalCaller {
protected:
    explicit MetalCaller(const config::BasecallModelConfig &model_config)
            : m_config(model_config) {}

public:
    static std::shared_ptr<MetalCaller> create(const config::BasecallModelConfig &model_config,
                                               float memory_limit_fraction);
    static int get_max_safe_batch_size(float memory_limit_fraction,
                                       const config::BasecallModelConfig &config);
    static int get_batch_size_granularity(const config::BasecallModelConfig &config);

    virtual ~MetalCaller();

    virtual at::Tensor create_input_tensor() const = 0;
    void call_chunks(at::Tensor &input,
                     int num_chunks,
                     std::vector<decode::DecodedChunk> &out_chunks);

    void terminate();
    void restart();

    const config::BasecallModelConfig &config() const { return m_config; }

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

    decode::DecoderOptions m_decoder_options;

    NS::SharedPtr<MTL::Device> m_device;
};

}  // namespace dorado::basecall
