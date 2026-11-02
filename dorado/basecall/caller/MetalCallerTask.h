#pragma once

#include "CallerModel.h"

#include <ATen/core/TensorBody.h>

#include <cstdint>

namespace dorado::basecall {

struct MetalCallerTask final : InFlightBatch {
    MetalCallerTask(at::Tensor *input_, int num_chunks_) : input(input_), num_chunks(num_chunks_) {}

    // LSTM: NTC - Tx: NCT
    at::Tensor *input;
    int num_chunks;
    // Basecall model recurrent output [N, T, C], only set when the model feeds modbase models.
    at::Tensor signal_embeddings;
    // Event ID to be signalled when decoding for this task is complete, set by
    // MetalCallerModel::submit.
    uint64_t decode_complete_event_id = static_cast<uint64_t>(0);
};

}  // namespace dorado::basecall
