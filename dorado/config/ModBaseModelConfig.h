#pragma once

#include "utils/types.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace dorado::config {

enum class ModelType : std::uint8_t { CONV_LSTM_V1, CONV_LSTM_V2, CONV_V1, UNKNOWN };
std::string to_string(const ModelType& model_type);
ModelType model_type_from_string(const std::string& model_type);
ModelType get_modbase_model_type(const std::filesystem::path& path);

bool is_modbase_model(const std::filesystem::path& path);

struct ModelGeneralParams {
    const ModelType model_type;
    const int size;
    const int kmer_len;
    const int num_out;
    const int stride;

    ModelGeneralParams(ModelType model_type_, int size_, int kmer_len_, int num_out_, int stride_);
};

struct LinearParams {
    const int in;
    const int out;

    LinearParams(int in_, int out_);
};

struct RefinementParams {
    const bool do_rough_rescale;  ///< Whether to perform rough rescaling
    const size_t center_idx;      ///< The position in the kmer at which to check the levels

    RefinementParams() : do_rough_rescale(false), center_idx(0) {}
    RefinementParams(int center_idx_);
};

struct ModificationParams {
    const std::vector<std::string> codes;       ///< The modified bases codes (e.g 'h', 'm', CHEBI)
    const std::vector<std::string> long_names;  ///< The long names of the modified bases.
    const size_t count;                         ///< Number of mods

    const std::string motif;    ///< The motif to look for modified bases within.
    const size_t motif_offset;  ///< The position of the canonical base within the motif.

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
    const int64_t samples_before;  ///< Number of context signal samples before a context hit.
    const int64_t samples_after;   ///< Number of context signal samples after a context hit.
    const int64_t samples;         ///< The total context samples (before + after)
    const int64_t chunk_size;      ///< The total samples in a chunk

    const int bases_before;  ///< Number of bases before the primary base of a kmer.
    const int bases_after;   ///< Number of bases after the primary base of a kmer.
    const int kmer_len;      ///< The kmer length given by `bases_before + bases_after + 1`

    const bool reverse;             ///< Reverse model data before processing (rna model)
    const bool base_start_justify;  ///< Justify the kmer encoding to start the context hit

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

// Determine the modbase alphabet from parameters and calculate offset positions for the results
ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params);

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models);

}  // namespace dorado::config
