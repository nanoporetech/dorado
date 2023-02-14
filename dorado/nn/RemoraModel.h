#pragma once

#include "modbase/remora_scaler.h"

#ifndef __APPLE__
#include <c10/cuda/CUDAStream.h>
#endif
#include <torch/torch.h>

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dorado {

namespace utils {
struct BaseModInfo;
}

class RemoraEncoder;
class RemoraScaler;

torch::nn::ModuleHolder<torch::nn::AnyModule> load_remora_model(
        const std::filesystem::path& model_path,
        torch::TensorOptions options);

struct BaseModParams {
    std::vector<std::string> mod_long_names;  ///< The long names of the modified bases.
    std::string motif;                        ///< The motif to look for modified bases within.
    size_t base_mod_count;                    ///< The number of modifications for the base.
    size_t motif_offset;  ///< The position of the canonical base within the motif.
    size_t context_before;  ///< The number of context samples in the signal the network looks at around a candidate base.
    size_t context_after;  ///< The number of context samples in the signal the network looks at around a candidate base.
    int bases_before;  ///< The number of bases before the primary base of a kmer.
    int bases_after;   ///< The number of bases after the primary base of a kmer.
    int offset;
    std::string mod_bases;
    std::vector<float> refine_kmer_levels;  ///< Expected kmer levels for rough rescaling
    size_t refine_kmer_len;                 ///< Length of the kmers for the specified kmer_levels
    size_t refine_kmer_center_idx;  ///< The position in the kmer at which to check the levels
    bool refine_do_rough_rescale;   ///< Whether to perform rough rescaling
};

class RemoraCaller {
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    torch::TensorOptions m_options;
    torch::Tensor m_input_sigs;
    torch::Tensor m_input_seqs;
#ifndef __APPLE__
    c10::optional<c10::Stream> m_stream;
#endif
    std::unique_ptr<RemoraScaler> m_scaler;

    BaseModParams m_params;
    const int m_batch_size;

public:
    RemoraCaller(const std::filesystem::path& model_path,
                 const std::string& device,
                 int batch_size,
                 size_t block_stride);
    const BaseModParams& params() const { return m_params; }

    torch::Tensor scale_signal(torch::Tensor signal,
                               const std::vector<int>& seq_ints,
                               const std::vector<uint64_t>& seq_to_sig_map) const;
    std::vector<size_t> get_motif_hits(const std::string& seq) const;

    void accept_chunk(int num_chunks, at::Tensor signal, const std::vector<float>& kmers);
    torch::Tensor call_chunks(int num_chunks);
};

}  // namespace dorado