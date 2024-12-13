#pragma once

#include "utils/bam_utils.h"
#include "utils/dev_utils.h"
#include "utils/modbase_parameters.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dorado::modbase {

using ModelType = utils::modbase::ModelType;

struct ModelGeneralParams {
    const ModelType model_type{ModelType::UNKNOWN};
    const int size{0};
    const int kmer_len{0};
    const int num_out{0};
    const int stride{0};

    ModelGeneralParams(ModelType model_type_, int size_, int kmer_len_, int num_out_, int stride_);
};

struct LinearParams {
    const int in{0};
    const int out{0};

    LinearParams(int in_, int out_);
};

struct RefinementParams {
    const bool do_rough_rescale{false};  ///< Whether to perform rough rescaling
    const size_t kmer_len;               ///< Length of the kmers for the specified kmer_levels
    const size_t center_idx;             ///< The position in the kmer at which to check the levels
    const std::vector<float> levels;     ///< Expected kmer levels for rough rescaling

    RefinementParams() : do_rough_rescale(false), kmer_len(0), center_idx(0), levels({}) {}
    RefinementParams(const int kmer_len_,
                     const int center_idx_,
                     std::vector<float> refine_kmer_levels_);
};

struct ModificationParams {
    const std::vector<std::string> codes;       ///< The modified bases codes (e.g 'h', 'm', CHEBI)
    const std::vector<std::string> long_names;  ///< The long names of the modified bases.
    const size_t count;                         ///< Number of mods

    const std::string motif;      ///< The motif to look for modified bases within.
    const size_t motif_offset{};  ///< The position of the canonical base within the motif.

    const char base;    ///< The canonical base 'ACGT'
    const int base_id;  ///< The canonical base id 0-3

    ModificationParams(std::vector<std::string> codes_,
                       std::vector<std::string> long_names_,
                       std::string motif_,
                       const size_t motif_offset_);

private:
    static char get_canonical_base_name(const std::string& motif, size_t motif_offset);
};

struct ContextParams {
    const int64_t samples_before{0};  ///< Number of context signal samples before a context hit.
    const int64_t samples_after{1};   ///< Number of context signal samples after a context hit.
    const int64_t samples{1};         ///< The total context samples (before + after)
    const int64_t chunk_size{1};      ///< The total samples in a chunk

    const int bases_before{0};  ///< Number of bases before the primary base of a kmer.
    const int bases_after{1};   ///< Number of bases after the primary base of a kmer.
    const int kmer_len{1};      ///< The kmer length given by `bases_before + bases_after + 1`

    const bool reverse{false};             ///< Reverse model data before processing (rna model)
    const bool base_start_justify{false};  ///< Justify the kmer encoding to start the context hit

    ContextParams(int64_t samples_before_,
                  int64_t samples_after_,
                  int64_t chunk_size_,
                  int bases_before_,
                  int bases_after_,
                  bool reverse_,
                  bool base_start_justify_);

    // Normalise `v` by `stride` strictly increasing the if needed.
    static int64_t normalise(const int64_t v, const int64_t stride);
    // Return the context params but normalised by a stride
    ContextParams normalised(const int stride) const;
};

struct ModBaseModelConfig {
    std::filesystem::path model_path;  ///< Path to modbase model
    ModelGeneralParams general;        ///< General model params for legacy model architectures
    ModificationParams mods;           ///< Params for the modifications being detected
    ContextParams context;             ///< Params for the context over which mods are inferred
    RefinementParams refine;           ///< Params for kmer refinement

    // Returns true if this modbase model processes chunks instead of context hits
    bool is_chunked_input_model() const { return general.model_type == ModelType::CONV_LSTM_V2; };

    ModBaseModelConfig(std::filesystem::path model_path_,
                       ModelGeneralParams general_,
                       ModificationParams mods_,
                       ContextParams context_,
                       RefinementParams refine_);
};

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path);

namespace test {
ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path,
                                             const std::vector<float>& test_kmer_levels);
}

// Determine the modbase alphabet from parameters and calculate offset positions for the results
ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params);

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models);

}  // namespace dorado::modbase
