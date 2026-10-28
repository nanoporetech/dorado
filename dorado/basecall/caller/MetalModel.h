#pragma once

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

struct MetalCallerTask;
namespace decode {
struct DecodedChunk;
}

using DecodedData = std::tuple<std::string, std::string, std::vector<uint8_t>>;

class MetalModel {
public:
    virtual ~MetalModel() = default;

    virtual at::Tensor create_input_tensor() const = 0;
    virtual bool call_task(MetalCallerTask& task,
                           std::mutex& inter_caller_mutex,
                           int try_count) = 0;
    virtual DecodedData decode(int chunk_idx) const = 0;
    virtual MTL::SharedEvent* decode_complete_event() const = 0;
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
    MetalModel() = default;

private:
    // Incrementing ID used to prevent the linear layer of run i+1 overwriting the scores of
    // run i before the CPU has finished decoding all run i's chunks. Start at 1, since at event
    // creation ID 0 is deemed to have been signalled.
    uint64_t m_next_decode_complete_event_id{1};
};

}  // namespace dorado::basecall
