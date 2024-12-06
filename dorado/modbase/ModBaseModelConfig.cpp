#include "ModBaseModelConfig.h"

#include "torch_utils/tensor_utils.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>
#include <toml.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Indicates that a value has no default and is therefore required
constexpr std::optional<int> REQUIRED = std::nullopt;

// Get an integer value from a toml::value asserting that it is within a closed interval.
// If no default is given then the key must exist in the toml::value.
int get_int_in_range(const toml::value& p,
                     const std::string& key,
                     int min_val,
                     int max_val,
                     std::optional<int> default_val) {
    const int val = default_val.has_value()
                            ? toml::find_or<int>(p, key, std::forward<int>(default_val.value()))
                            : toml::find<int>(p, key);
    if (val < min_val || val > max_val) {
        auto v = std::to_string(val);
        auto r = std::to_string(min_val) + " <= x <= " + std::to_string(max_val);
        throw std::runtime_error("Invalid modbase model value for '" + key + "' found: '" + v +
                                 "' which is not in range [" + r + "]");
    }
    return val;
}
}  // namespace

namespace dorado::modbase {

std::string to_string(const ModelType& model_type) {
    switch (model_type) {
    case CONV_LSTM_V1:
        return std::string("conv_lstm");
    case CONV_LSTM_V2:
        return std::string("conv_lstm_v2");
    case CONV_V1:
        return std::string("conv_v1");
    default:
        throw std::runtime_error("Unknown modbase ModelType");
    }
};

ModelType model_type_from_string(const std::string& model_type) {
    if (model_type == "conv_lstm") {
        return ModelType::CONV_LSTM_V1;
    }
    if (model_type == "conv_lstm_v2") {
        return ModelType::CONV_LSTM_V2;
    }
    if (model_type == "conv_only" || model_type == "conv_v1") {
        return ModelType::CONV_V1;
    }
    throw std::runtime_error("Unknown modbase model type: `" + model_type + "`");
}

ModelType parse_model_type(const toml::value& config_toml) {
    return model_type_from_string(toml::find<std::string>(config_toml, "general", "model"));
}

ModelGeneralParams parse_general_params(const toml::value& config_toml) {
    const auto segment = toml::find(config_toml, "model_params");

    constexpr int MAX_SIZE = 4096;
    constexpr int MAX_KMER = 19;
    constexpr int MAX_FEATURES = 10;
    constexpr int MAX_STRIDE = 6;
    ModelGeneralParams params{
            parse_model_type(config_toml),
            get_int_in_range(segment, "size", 1, MAX_SIZE, REQUIRED),
            get_int_in_range(segment, "kmer_len", 1, MAX_KMER, REQUIRED),
            get_int_in_range(segment, "num_out", 1, MAX_FEATURES, REQUIRED),
            get_int_in_range(segment, "stride", 1, MAX_STRIDE, 3),
    };
    return params;
}

LinearParams parse_linear_params(const toml::value& segment) {
    constexpr int MAX_SIZE = 4096;
    LinearParams params{get_int_in_range(segment, "in", 1, MAX_SIZE, REQUIRED),
                        get_int_in_range(segment, "out", 1, MAX_SIZE, REQUIRED)};
    return params;
}

ModificationParams parse_modification_params(const toml::value& config_toml) {
    const auto& params = toml::find(config_toml, "modbases");

    std::vector<std::string> codes;
    const auto& mod_bases = toml::find(params, "mod_bases");
    if (mod_bases.is_string()) {
        // style: mod_bases = "hm" - does not accept chebi codes
        auto mod_base_string = mod_bases.as_string().str;
        for (const auto& mod_base : mod_base_string) {
            codes.push_back(std::string(1, mod_base));
        }
    } else {
        // style: mod_bases = [ "h", "m",]
        auto mod_base_array = mod_bases.as_array();
        for (const auto& mod_base : mod_base_array) {
            assert(mod_base.is_string());
            codes.push_back(mod_base.as_string().str);
        }
    }

    std::vector<std::string> long_names;
    long_names.reserve(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    const auto motif = toml::find<std::string>(params, "motif");
    const auto motif_offset = static_cast<size_t>(
            get_int_in_range(params, "motif_offset", 0, int(motif.size()), REQUIRED));

    return ModificationParams{std::move(codes), std::move(long_names), motif, motif_offset};
}

ContextParams parse_context_params(const toml::value& config_toml) {
    const auto& params = toml::find(config_toml, "modbases");

    const int context_before = get_int_in_range(params, "chunk_context_0", 0, 4096, REQUIRED);
    const int context_after = get_int_in_range(params, "chunk_context_1", 1, 4096, REQUIRED);

    constexpr int MAX_CHUNK_SIZE = 102400;
    const int min_chunk_size = context_before + context_after;
    const int chunk_size =
            get_int_in_range(params, "chunk_size", min_chunk_size, MAX_CHUNK_SIZE, min_chunk_size);

    const auto bases_before = get_int_in_range(params, "kmer_context_bases_0", 0, 9, REQUIRED);
    const auto bases_after = get_int_in_range(params, "kmer_context_bases_1", 0, 9, REQUIRED);

    const bool reverse = toml::find_or<bool>(params, "reverse_signal", false);
    const bool base_start_justify = toml::find_or<bool>(params, "base_start_justify", false);

    return ContextParams(context_before, context_after, chunk_size, bases_before, bases_after,
                         reverse, base_start_justify);
}

ContextParams ContextParams::normalised(const int stride) const {
    const int64_t sb = normalise(samples_before, stride);
    const int64_t sa = normalise(samples_after, stride);
    const int64_t cs = normalise(chunk_size, stride);

    spdlog::trace(
            "Normalised modbase context for stride: {} - [{}, {}, {}] -> [{}, {}, {}] @ "
            "[samples_before, samples_after, chunk_size]",
            stride, samples_before, samples_after, chunk_size, sb, sa, cs);
    return ContextParams(sb, sa, cs, bases_before, bases_after, reverse, base_start_justify);
}

RefinementParams parse_refinement_params(const toml::value& config_toml,
                                         const std::filesystem::path& model_path,
                                         const std::vector<float>& _test_levels) {
    if (!config_toml.contains("refinement")) {
        return RefinementParams{};
    }

    const auto segment = toml::find(config_toml, "refinement");

    bool do_rough_rescale = toml::find<int>(segment, "refine_do_rough_rescale") == 1;
    if (!do_rough_rescale) {
        return RefinementParams{};
    }

    std::vector<float> kmer_levels;
    // allow us to avoid adding tensors to the repo just for testing
    if (_test_levels.empty()) {
        auto kmer_levels_tensor =
                utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0].contiguous();
        std::copy(kmer_levels_tensor.data_ptr<float>(),
                  kmer_levels_tensor.data_ptr<float>() + kmer_levels_tensor.numel(),
                  std::back_inserter(kmer_levels));
    } else {
        kmer_levels = _test_levels;
    }

    const int kmer_len = static_cast<int>(std::round(std::log(kmer_levels.size()) / std::log(4)));
    const int center_index =
            get_int_in_range(segment, "refine_kmer_center_idx", 0, kmer_len - 1, REQUIRED);

    return RefinementParams(kmer_len, center_index, std::move(kmer_levels));
}

ModBaseModelConfig load_modbase_model_config_impl(const std::filesystem::path& model_path,
                                                  const std::vector<float>& test_kmer_levels) {
    const auto config_toml = toml::parse(model_path / "config.toml");

    ModBaseModelConfig config{model_path, parse_general_params(config_toml),
                              parse_modification_params(config_toml),
                              parse_context_params(config_toml),
                              parse_refinement_params(config_toml, model_path, test_kmer_levels)};

    return config;
}

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path) {
    return load_modbase_model_config_impl(model_path, {});
}

namespace test {
ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path,
                                             const std::vector<float>& test_kmer_levels) {
    return load_modbase_model_config_impl(model_path, test_kmer_levels);
}
}  // namespace test

ModBaseInfo get_modbase_info(
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::vector<std::string> alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    const std::string allowed_bases = "ACGT";
    std::array<ModelInfo, utils::BaseInfo::NUM_BASES> model_info;
    for (int b = 0; b < utils::BaseInfo::NUM_BASES; ++b) {
        model_info[b].alphabet.emplace_back(1, allowed_bases[b]);
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = params_ref.get().mods;
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in modbase model metadata.");
        }
        auto& map_entry = model_info[utils::BaseInfo::BASE_IDS[base]];
        map_entry.long_names = params.long_names;
        map_entry.alphabet.insert(map_entry.alphabet.end(), params.codes.begin(),
                                  params.codes.end());
        map_entry.base_counts = params.count + 1;
    }

    ModBaseInfo result;
    size_t index = 0;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!result.long_names.empty()) {
                result.long_names += ' ';
            }
            result.long_names += name;
        }
        result.alphabet.insert(result.alphabet.end(), info.alphabet.begin(), info.alphabet.end());
        result.base_counts[index++] = info.base_counts;
    }

    return result;
}

void check_modbase_multi_model_compatibility(
        const std::vector<std::filesystem::path>& modbase_models) {
    if (modbase_models.size() < 2) {
        return;
    }

    std::string err_msg = "";
    for (size_t i = 0; i < modbase_models.size(); i++) {
        const auto ref_model = load_modbase_model_config(modbase_models[i]);
        const auto& ref_params = ref_model.mods;
        const auto& ref_motif = ref_params.motif[ref_params.motif_offset];

        for (size_t j = i + 1; j < modbase_models.size(); j++) {
            const auto query_model = load_modbase_model_config(modbase_models[j]);
            const auto& query_params = query_model.mods;
            const auto& query_motif = query_params.motif[query_params.motif_offset];

            if (ref_motif == query_motif) {
                err_msg += modbase_models[i].string() + " and " + modbase_models[j].string() +
                           " have overlapping canonical motif: " + ref_motif;
            }
        }
    }

    if (!err_msg.empty()) {
        throw std::runtime_error(
                "Following are incompatible modbase models. Please select only one of them to "
                "run:\n" +
                err_msg);
    }
}

}  // namespace dorado::modbase
