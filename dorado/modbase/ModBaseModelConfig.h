#pragma once

#include "utils/types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace dorado::modbase {

struct ModBaseModelConfig {
    std::vector<std::string> mod_long_names;  ///< The long names of the modified bases.
    std::string motif;                        ///< The motif to look for modified bases within.
    size_t base_mod_count{};                  ///< The number of modifications for the base.
    size_t motif_offset{};     ///< The position of the canonical base within the motif.
    size_t context_samples{};  ///< The context samples given by `context_before + context_after`
    size_t context_before{};  ///< The number of context samples in the signal the network looks at around a candidate base.
    size_t context_after{};  ///< The number of context samples in the signal the network looks at around a candidate base.
    int kmer_length{};   ///< The kmer length given by `bases_before + bases_after + 1`
    int bases_before{};  ///< The number of bases before the primary base of a kmer.
    int bases_after{};   ///< The number of bases after the primary base of a kmer.
    int offset{};
    std::vector<std::string> mod_bases;
    std::vector<float> refine_kmer_levels;  ///< Expected kmer levels for rough rescaling
    size_t refine_kmer_len{};               ///< Length of the kmers for the specified kmer_levels
    size_t refine_kmer_center_idx{};      ///< The position in the kmer at which to check the levels
    bool refine_do_rough_rescale{false};  ///< Whether to perform rough rescaling
    bool reverse_signal{false};           ///< Reverse model data before processing (rna model)
};

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path);

// Determine the modbase alphabet from parameters and calculate offset positions for the results
ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params);

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models);

}  // namespace dorado::modbase
