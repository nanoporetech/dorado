#pragma once

#include "utils/types.h"

#include <filesystem>
#include <string>
#include <vector>

namespace dorado {

struct ModBaseModelConfig {
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

ModBaseModelConfig load_modbase_model_config(std::filesystem::path const& model_path);

// Determine the modbase alphabet from parameters and calculate offset positions for the results
ModBaseInfo get_modbase_info(
        std::vector<std::reference_wrapper<ModBaseModelConfig const>> const& base_mod_params);

}  // namespace dorado
