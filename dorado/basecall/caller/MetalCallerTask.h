#pragma once

#include "basecall/DecodedChunk.h"

#include <ATen/core/TensorBody.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace dorado::basecall {

struct MetalCallerTask {
    MetalCallerTask(at::Tensor *input_,
                    int num_chunks_,
                    std::vector<decode::DecodedChunk> *out_chunks_)
            : input(input_), out_chunks(out_chunks_), num_chunks(num_chunks_) {}

    // LSTM: NTC - Tx: NCT
    at::Tensor *input;
    std::mutex mut;
    std::condition_variable cv;
    std::vector<decode::DecodedChunk> *out_chunks;
    int num_chunks;
    // Basecall model recurrent output [N, T, C], only set when the model feeds modbase models.
    at::Tensor signal_embeddings;
    int decode_chunks_started{0};
    int decode_chunks_finished{0};
    // Set once all chunks are decoded and any modbase results are populated.
    bool done{false};
    // Event ID to be signalled when decoding for this task is complete, set by metal_thread_fn.
    uint64_t decode_complete_event_id = static_cast<uint64_t>(0);
};

}  // namespace dorado::basecall
