#pragma once

#include "utils/stats.h"

#include <ATen/core/TensorBody.h>
#if DORADO_GPU_BUILD && !defined(__APPLE__)
#include <c10/cuda/CUDAStream.h>
#endif
#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::modbase {

struct ModBaseModelConfig;
class ModBaseCaller;

std::shared_ptr<ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device);

class ModBaseRunner {
public:
    explicit ModBaseRunner(std::shared_ptr<ModBaseCaller> caller);
    void accept_chunk(int model_id,
                      int chunk_idx,
                      const at::Tensor& signal,
                      const std::vector<int8_t>& kmers);
    at::Tensor call_chunks(int model_id, int num_chunks);
    at::Tensor scale_signal(size_t caller_id,
                            at::Tensor signal,
                            const std::vector<int>& seq_ints,
                            const std::vector<uint64_t>& seq_to_sig_map) const;
    std::vector<size_t> get_motif_hits(size_t caller_id, const std::string& seq) const;
    const ModBaseModelConfig& caller_params(size_t caller_id) const;
    size_t num_callers() const;
    size_t batch_size() const { return m_input_sigs[0].size(0); }
    void terminate();
    void restart();
    std::string get_name() const;
    stats::NamedStats sample_stats() const;

private:
    std::shared_ptr<ModBaseCaller> m_caller;
    std::vector<at::Tensor> m_input_sigs;
    std::vector<at::Tensor> m_input_seqs;
    std::vector<c10::optional<c10::Stream>> m_streams;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

using RunnerPtr = std::unique_ptr<ModBaseRunner>;

}  // namespace dorado::modbase
