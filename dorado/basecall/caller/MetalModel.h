#pragma once

#include "config/BasecallModelConfig.h"
#include "torch_utils/metal_utils.h"
#include "utils/types.h"

#include <ATen/core/TensorBody.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace dorado::basecall {

class MetalDecoder;
struct MetalCallerTask;
namespace decode {
struct DecodedChunk;
}

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalModel {
public:
    virtual ~MetalModel();

    virtual at::Tensor create_input_tensor() const = 0;
    virtual bool call_task(MetalCallerTask& task,
                           std::mutex& inter_caller_mutex,
                           int try_count) = 0;
    DecodedData decode(int chunk_idx) const;
    MTL::SharedEvent* decode_complete_event() const { return m_decode_complete_event.get(); }
    // Models that support modbase calling override these. decode_modbase is called once all of a
    // task's chunks have been decoded.
    virtual std::shared_ptr<ModBaseInfo> modbase_info() const { return nullptr; }
    virtual void decode_modbase(MetalCallerTask& task, std::span<decode::DecodedChunk> chunks) {
        (void)task;
        (void)chunks;
    }

    // Only to be called from the worker thread.
    uint64_t next_decode_complete_event_id() { return m_next_decode_complete_event_id++; }

protected:
    explicit MetalModel(const config::BasecallModelConfig& model_config);

    const config::BasecallModelConfig m_config;
    NS::SharedPtr<MTL::Device> m_device;

    // Set by the derived class.
    int m_in_chunk_size{0}, m_out_chunk_size{0}, m_batch_size{0};
    std::unique_ptr<MetalDecoder> m_decoder;

private:
    NS::SharedPtr<MTL::SharedEvent> m_decode_complete_event;

    // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
    // run i before the CPU has finished decoding all run i's chunks. Start at 1, since at event
    // creation ID 0 is deemed to have been signalled.
    uint64_t m_next_decode_complete_event_id{1};
};

}  // namespace dorado::basecall
