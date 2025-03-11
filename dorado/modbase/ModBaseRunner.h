#pragma once

#include "utils/stats.h"

#include <ATen/core/TensorBody.h>
#if DORADO_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>
#endif
#include <atomic>
#include <string>
#include <vector>

namespace dorado::config {
struct ModBaseModelConfig;
}

namespace dorado::modbase {

class ModBaseCaller;

class ModBaseRunner {
public:
    explicit ModBaseRunner(std::shared_ptr<ModBaseCaller> caller);
    // Copy the signal and kmer data `chunk_idx` into the pending batch at the same index.
    void accept_chunk(int model_id,
                      int chunk_idx,
                      const at::Tensor& signal,
                      const std::vector<int8_t>& kmers);
    // Call enqueued chunks
    at::Tensor call_chunks(int model_id, int num_chunks);
    // Scale the signal tensor for the modbase model
    at::Tensor scale_signal(size_t model_id,
                            at::Tensor signal,
                            const std::vector<int>& seq_ints,
                            const std::vector<uint64_t>& seq_to_sig_map) const;
    // Get the sequence indexes of all motif hits for this sequence
    std::vector<size_t> get_motif_hits(size_t model_id, const std::string& seq) const;
    // Get the modbase model config
    const config::ModBaseModelConfig& model_params(size_t model_id) const;
    // Get the integer base_id for a model
    int model_base_id(size_t model_id) const;
    // The number of modbase models that have been loaded
    size_t num_models() const;
    // FIXME: This might need to be independent of the runner
    // The number of chunks in a batch
    size_t batch_size() const { return m_input_sigs[0].size(0); }

    // Asserts all modbase models are either chunked or context-centered
    bool is_chunked_model_type() const;
    // Only meaningful for models accepting chunked inputs
    bool takes_chunk_inputs() const;

    void terminate();
    void restart();
    std::string get_name() const;
    stats::NamedStats sample_stats() const;

private:
    std::shared_ptr<ModBaseCaller> m_caller;
    // Contains signal chunks: N(batch_size) C(1) T(chunk_size)
    std::vector<at::Tensor> m_input_sigs;
    // Contains encoded Kmers: N(batch_size) T(chunk_size), C(Kmer*Bases)
    std::vector<at::Tensor> m_input_seqs;

    const bool m_is_chunked_model_type;

#if DORADO_CUDA_BUILD
    std::vector<c10::optional<c10::Stream>> m_streams;
#endif
    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

using RunnerPtr = std::unique_ptr<ModBaseRunner>;

}  // namespace dorado::modbase
