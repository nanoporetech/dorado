#pragma once

#include "utils/stats.h"

#include <torch/torch.h>

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

class ModBaseCaller;

struct ModBaseParams {
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

    void parse(std::filesystem::path const& model_path, bool all_members = true);
};

std::shared_ptr<ModBaseCaller> create_modbase_caller(
        const std::vector<std::filesystem::path>& model_paths,
        int batch_size,
        const std::string& device);

class ModBaseRunner {
public:
    explicit ModBaseRunner(std::shared_ptr<ModBaseCaller> caller);
    void accept_chunk(int model_id,
                      int chunk_idx,
                      const torch::Tensor& signal,
                      const std::vector<int8_t>& kmers);
    torch::Tensor call_chunks(int model_id, int num_chunks);
    torch::Tensor scale_signal(size_t caller_id,
                               torch::Tensor signal,
                               const std::vector<int>& seq_ints,
                               const std::vector<uint64_t>& seq_to_sig_map) const;
    std::vector<size_t> get_motif_hits(size_t caller_id, const std::string& seq) const;
    ModBaseParams& caller_params(size_t caller_id) const;
    size_t num_callers() const;
    void terminate();
    std::string get_name() const;
    stats::NamedStats sample_stats() const;

private:
    std::shared_ptr<ModBaseCaller> m_caller;
    std::vector<torch::Tensor> m_input_sigs;
    std::vector<torch::Tensor> m_input_seqs;

    // Performance monitoring stats.
    std::atomic<int64_t> m_num_batches_called = 0;
};

}  // namespace dorado
