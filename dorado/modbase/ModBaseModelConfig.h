#pragma once

#include "utils/bam_utils.h"
#include "utils/types.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
const std::string err = "Invalid modbase model parameter in ";
}

namespace dorado::modbase {

enum ModelType { CONV_LSTM, CONV_V1, CONV_V2, UNKNOWN };
std::string to_string(const ModelType& model_type);
ModelType model_type_from_string(const std::string& model_type);

struct ModelGeneralParams {
    const ModelType model_type{ModelType::UNKNOWN};
    const int size{0};
    const int kmer_len{0};
    const int num_out{0};

    ModelGeneralParams(ModelType model_type_, int size_, int kmer_len_, int num_out_)
            : model_type(model_type_), size(size_), kmer_len(kmer_len_), num_out(num_out_) {
        if (model_type == ModelType::UNKNOWN) {
            throw std::runtime_error(err + "general params: 'model type not set or unknown'");
        }
        if (model_type == ModelType::CONV_V2) {
            throw std::runtime_error(err + "general params: 'conv_v2 models not implemented'");
        }
        if (size < 1 || kmer_len < 1 || num_out < 1) {
            throw std::runtime_error(err + "general params: 'negative or zero value'.");
        }
        if (kmer_len % 2 != 1) {
            throw std::runtime_error(err + "general params: 'kmer_length is not odd'.");
        }
    }
};

struct LinearParams {
    const int in{0};
    const int out{0};

    LinearParams(int in_, int out_) : in(in_), out(out_) {
        if (in < 1 || out < 1) {
            throw std::runtime_error(err + "linear params: 'negative or zero value'.");
        }
    }
};

struct ConvParams {
    const int insize;
    const int size;
    const int winlen;
    const int stride;
    const int depth;
    const int padding;

    ConvParams(int insize_, int size_, int winlen_, int stride_, int depth_, int padding_)
            : insize(insize_),
              size(size_),
              winlen(winlen_),
              stride(stride_),
              depth(depth_),
              padding(padding_) {
        if (insize < 1 || size < 1 || winlen < 1 || stride < 1 || depth < 1 || padding < 1) {
            throw std::runtime_error(err + "conv params: 'negative or zero value'.");
        }
        if (depth > 1 && (insize != size)) {
            throw std::runtime_error(
                    err + "conv params: 'repeated convs (depth > 1) must not change size'.");
        }
    };
};

struct ConvEncoder {
    const std::vector<ConvParams> stem;
    const std::vector<ConvParams> downsample;
    const std::vector<ConvParams> stages;
    const LinearParams head;

    ConvEncoder(std::vector<ConvParams> stem_,
                std::vector<ConvParams> downsample_,
                std::vector<ConvParams> stages_,
                const LinearParams& head_)
            : stem(std::move(stem_)),
              downsample(std::move(downsample_)),
              stages(std::move(stages_)),
              head(head_) {
        if (stem.empty()) {
            throw std::runtime_error(err + "conv encoder: 'missing stem'.");
        }
        if (downsample.empty()) {
            throw std::runtime_error(err + "conv encoder: 'missing downsample'.");
        }
        if (stages.empty()) {
            throw std::runtime_error(err + "conv encoder: 'missing stages'.");
        }
    };
};

struct RefinementParams {
    const bool do_rough_rescale{false};  ///< Whether to perform rough rescaling
    const size_t kmer_len;               ///< Length of the kmers for the specified kmer_levels
    const size_t center_idx;             ///< The position in the kmer at which to check the levels
    const std::vector<float> levels;     ///< Expected kmer levels for rough rescaling

    RefinementParams() : do_rough_rescale(false), kmer_len(0), center_idx(0) {}
    RefinementParams(const int kmer_len_,
                     const int center_idx_,
                     std::vector<float> refine_kmer_levels_)
            : do_rough_rescale(true),
              kmer_len(static_cast<size_t>(kmer_len_)),
              center_idx(static_cast<size_t>(center_idx_)),
              levels(std::move(refine_kmer_levels_)) {
        if (kmer_len < 1 || kmer_len_ < 1) {
            throw std::runtime_error(err + "refinement params: 'negative or zero kmer len'.");
        }
        if (center_idx_ < 0) {
            throw std::runtime_error(err + "refinement params: 'negative center index'.");
        }
        if (kmer_len < center_idx) {
            throw std::runtime_error(err + "refinement params: 'invalid center index'.");
        }
        if (levels.empty()) {
            throw std::runtime_error(err + "refinement params: 'missing levels'.");
        }
    }
};

struct ModificationParams {
    const std::vector<std::string> codes;       ///< The modified bases codes (e.g 'h', 'm', CHEBI)
    const std::vector<std::string> long_names;  ///< The long names of the modified bases.
    const size_t count;                         ///< The number of mods

    const std::string motif;      ///< The motif to look for modified bases within.
    const size_t motif_offset{};  ///< The position of the canonical base within the motif.

    ModificationParams(std::vector<std::string> codes_,
                       std::vector<std::string> long_names_,
                       std::string motif_,
                       const size_t motif_offset_)
            : codes(std::move(codes_)),
              long_names(std::move(long_names_)),
              count(codes.size()),
              motif(std::move(motif_)),
              motif_offset(motif_offset_) {
        if (motif.size() < motif_offset) {
            throw std::runtime_error(err + "mods params: 'invalid motif offset'.");
        }

        const std::string canonical_bases = "ACGT";
        std::string motif_base = motif.substr(motif_offset, 1);
        if (canonical_bases.find(motif_base) == std::string::npos) {
            throw std::runtime_error(err + "mods params: 'invalid motif base " + motif_base + "'.");
        }

        if (codes.empty()) {
            throw std::runtime_error(err + "mods params: 'empty modifications.");
        }
        if (long_names.empty()) {
            throw std::runtime_error(err + "mods params: 'empty long names.");
        }
        if (codes.size() != long_names.size()) {
            throw std::runtime_error(err + "mods params: 'mods and names size mismatch.");
        }

        for (const auto& code : codes) {
            if (!utils::validate_bam_tag_code(code)) {
                std::string e = err + "mods params: 'invalid mod code ";
                throw std::runtime_error(e + code + "'.");
            }
        }
    }
};

struct ContextParams {
    const size_t samples_before;  ///< The number of context signal samples before a candidate base.
    const size_t samples_after;   ///< The number of context signal samples after a candidate base.
    const size_t samples;         ///< The total context samples (before + after)

    const int bases_before;  ///< The number of bases before the primary base of a kmer.
    const int bases_after;   ///< The number of bases after the primary base of a kmer.
    const int kmer_len;      ///< The kmer length given by `bases_before + bases_after + 1`

    const bool reverse;             ///< Reverse model data before processing (rna model)
    const bool base_start_justify;  ///< To justify the kmer encoding to start the context hit

    ContextParams(size_t samples_before_,
                  size_t samples_after_,
                  int bases_before_,
                  int bases_after_,
                  bool reverse_,
                  bool base_start_justify_)
            : samples_before(samples_before_),
              samples_after(samples_after_),
              samples(samples_before + samples_after),
              bases_before(bases_before_),
              bases_after(bases_after_),
              kmer_len(bases_before_ + bases_after_ + 1),
              reverse(reverse_),
              base_start_justify(base_start_justify_) {
        if (samples_before < 1 || samples_after < 1) {
            throw std::runtime_error(err + "context params: 'negative or zero context samples'.");
        }
        if (bases_before < 1 || bases_after < 1) {
            throw std::runtime_error(err + "context params: 'negative or zero context bases'.");
        }
    }
};

struct ModBaseModelConfig {
    std::filesystem::path model_path;  ///< Path to modbase model
    ModelGeneralParams general;        ///< General model params for legacy model architectures
    ModificationParams mods;           ///< Params for the modifications being detected
    ContextParams context;             ///< Params for the context over which mods are inferred
    RefinementParams refine;           ///< Params for kmer refinement

    ModBaseModelConfig(std::filesystem::path model_path_,
                       ModelGeneralParams general_,
                       ModificationParams mods_,
                       ContextParams context_,
                       RefinementParams refine_)
            : model_path(std::move(model_path_)),
              general(std::move(general_)),
              mods(std::move(mods_)),
              context(std::move(context_)),
              refine(std::move(refine_)) {
        // Kmer length is duplicated in modbase model configs - check they match
        if (general.kmer_len != context.kmer_len) {
            auto kl_a = std::to_string(general.kmer_len);
            auto kl_b = std::to_string(context.kmer_len);
            throw std::runtime_error(err + "config: 'inconsistent kmer_len: " + kl_a +
                                     " != " + kl_b + "'.");
        }
    }
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
