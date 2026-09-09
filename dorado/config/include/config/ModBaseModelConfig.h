#pragma once

#include "common.h"
#include "config/ModBaseModelConfigFwd.h"
#include "utils/types.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado::config {

enum class ModelType : std::uint8_t { CONV_LSTM_V1, CONV_LSTM_V2, CONV_LSTM_V3, CONV_V1, UNKNOWN };
std::string to_string(const ModelType& model_type);
ModelType model_type_from_string(const std::string& model_type);
ModelType get_modbase_model_type(const std::filesystem::path& path);

bool is_modbase_model(const std::filesystem::path& path);

struct LinearParams {
    int in_size;
    int out_size;
};

struct LSTMParams {
    int size;
    bool reverse;
};

struct ModulesParams {
    std::vector<ConvParams> sequence_convs;
    std::vector<ConvParams> signal_convs;
    ConvParams merge_conv;
    std::vector<LSTMParams> lstms;  //< LSTM sizes per layer
    LinearParams linear;
    std::optional<LinearUpsampleParams> upsample;

    int stride_ratio() const;
    int sequence_stride() const;
    int signal_stride() const;
};

struct CommonModelParams {
    const int size;
    const int kmer_len;
    const int num_out;
    const int stride;
    CommonModelParams(int size_, int kmer_len_, int num_out_, int stride_);
};

struct ChunkedModelParams {
    const int64_t chunk_size;
    explicit ChunkedModelParams(int64_t chunk_size_);
};

struct ConvLSTMV3ModelParams {
    const CommonModelParams common;
    const ChunkedModelParams chunked;
    const ModulesParams modules;
    const int sequence_stride;
    ConvLSTMV3ModelParams(CommonModelParams common_,
                          ChunkedModelParams chunked_,
                          int sequence_stride_,
                          ModulesParams modules_);
};

struct RefinementParams {
    const bool do_rough_rescale;  ///< Whether to perform rough rescaling
    const size_t center_idx;      ///< The position in the kmer at which to check the levels

    RefinementParams() : do_rough_rescale(false), center_idx(0) {}
    explicit RefinementParams(int center_idx_);
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

    const int bases_before;  ///< Number of bases before the primary base of a kmer.
    const int bases_after;   ///< Number of bases after the primary base of a kmer.
    const int kmer_len;      ///< The kmer length given by `bases_before + bases_after + 1`

    const bool reverse;             ///< Reverse model data before processing (rna model)
    const bool base_start_justify;  ///< Justify the kmer encoding to start the context hit

    ContextParams(int64_t samples_before_,
                  int64_t samples_after_,
                  int bases_before_,
                  int bases_after_,
                  bool reverse_,
                  bool base_start_justify_);

    // Return the context params but normalised by a stride
    ContextParams normalised(const int stride) const;
};

struct ModBaseModelConfigData {
    std::filesystem::path model_path;  ///< Path to modbase model
    ModificationParams mods;           ///< Params for the modifications being detected
    ContextParams context;             ///< Params for the context over which mods are inferred
    RefinementParams refine;           ///< Params for kmer refinement

    ModBaseModelConfigData(std::filesystem::path model_path_,
                           ModificationParams mods_,
                           ContextParams context_,
                           RefinementParams refine_,
                           int kmer_len_);
};

struct ConvOnlyModelConfig {
    ModBaseModelConfigData data;
    CommonModelParams general;
    ConvOnlyModelConfig(std::filesystem::path model_path_,
                        CommonModelParams general_,
                        ModificationParams mods_,
                        ContextParams context_,
                        RefinementParams refine_);
};

struct ConvLSTMModelConfig {
    ModBaseModelConfigData data;
    CommonModelParams general;
    ConvLSTMModelConfig(std::filesystem::path model_path_,
                        CommonModelParams general_,
                        ModificationParams mods_,
                        ContextParams context_,
                        RefinementParams refine_);
};

struct ConvLSTMV2ModelConfig {
    ModBaseModelConfigData data;
    CommonModelParams general;
    ChunkedModelParams chunked;
    ConvLSTMV2ModelConfig(std::filesystem::path model_path_,
                          CommonModelParams general_,
                          ChunkedModelParams chunked_,
                          ModificationParams mods_,
                          ContextParams context_,
                          RefinementParams refine_);
};

struct ConvLSTMV3ModelConfig {
    ModBaseModelConfigData data;
    ConvLSTMV3ModelParams general;
    ConvLSTMV3ModelConfig(std::filesystem::path model_path_,
                          ConvLSTMV3ModelParams general_,
                          ModificationParams mods_,
                          ContextParams context_,
                          RefinementParams refine_);
};

const ModBaseModelConfigData& get_model_config_data(const ModBaseModelConfig& config);
const CommonModelParams& get_common_model_params(const ModBaseModelConfig& config);
ModelType get_model_type(const ModBaseModelConfig& config);
bool is_chunked_input_model(const ModBaseModelConfig& config);

const ChunkedModelParams& get_chunked_model_params(const ModBaseModelConfig& config);
int get_chunked_model_stride_ratio(const ModBaseModelConfig& config);
std::tuple<int64_t, int64_t> get_chunked_sequence_input_TC(const ConvLSTMV3ModelConfig& config);
std::tuple<int64_t, int64_t> get_chunked_signal_input_TC(const ConvLSTMV3ModelConfig& config);
std::tuple<int64_t, int64_t> get_chunked_sequence_input_TC(const ModBaseModelConfig& config);
std::tuple<int64_t, int64_t> get_chunked_signal_input_TC(const ModBaseModelConfig& config);
std::tuple<int64_t, int64_t> get_chunked_output_TC(const ModBaseModelConfig& config);
const ModBaseModelConfigData& get_context_model_config(const ModBaseModelConfig& config);

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path);

// Determine the modbase alphabet from parameters and calculate offset positions for the results
ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params);

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models);

}  // namespace dorado::config
