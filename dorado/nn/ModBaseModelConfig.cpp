#include "ModBaseModelConfig.h"

#include "utils/sequence_utils.h"
#include "utils/tensor_utils.h"

#include <toml.hpp>

namespace dorado {

ModBaseModelConfig load_modbase_model_config(std::filesystem::path const& model_path) {
    ModBaseModelConfig config;
    auto config_toml = toml::parse(model_path / "config.toml");
    const auto& params = toml::find(config_toml, "modbases");
    config.motif = toml::find<std::string>(params, "motif");
    config.motif_offset = toml::find<int>(params, "motif_offset");

    config.mod_bases = toml::find<std::string>(params, "mod_bases");
    for (size_t i = 0; i < config.mod_bases.size(); ++i) {
        config.mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    config.base_mod_count = config.mod_long_names.size();

    config.context_before = toml::find<int>(params, "chunk_context_0");
    config.context_after = toml::find<int>(params, "chunk_context_1");
    config.bases_before = toml::find<int>(params, "kmer_context_bases_0");
    config.bases_after = toml::find<int>(params, "kmer_context_bases_1");
    config.offset = toml::find<int>(params, "offset");

    if (config_toml.contains("refinement")) {
        // these may not exist if we convert older models
        const auto& refinement_params = toml::find(config_toml, "refinement");
        config.refine_do_rough_rescale =
                (toml::find<int>(refinement_params, "refine_do_rough_rescale") == 1);
        if (config.refine_do_rough_rescale) {
            config.refine_kmer_center_idx =
                    toml::find<int>(refinement_params, "refine_kmer_center_idx");

            auto kmer_levels_tensor =
                    utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0].contiguous();
            std::copy(kmer_levels_tensor.data_ptr<float>(),
                      kmer_levels_tensor.data_ptr<float>() + kmer_levels_tensor.numel(),
                      std::back_inserter(config.refine_kmer_levels));
            config.refine_kmer_len = static_cast<size_t>(
                    std::round(std::log(config.refine_kmer_levels.size()) / std::log(4)));
        }

    } else {
        // if the toml file doesn't contain any of the above parameters then
        // the model doesn't support rescaling so turn it off
        config.refine_do_rough_rescale = false;
    }
    return config;
}

ModBaseInfo get_modbase_info(
        std::vector<std::reference_wrapper<ModBaseModelConfig const>> const& base_mod_params) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::string alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    std::string const allowed_bases = "ACGT";
    std::array<ModelInfo, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet = allowed_bases[b];
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = params_ref.get();
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in remora model metadata.");
        }
        auto& map_entry = model_info[utils::BaseInfo::BASE_IDS[base]];
        map_entry.long_names = params.mod_long_names;
        map_entry.alphabet += params.mod_bases;
        map_entry.base_counts = params.base_mod_count + 1;
    }

    ModBaseInfo result;
    size_t index = 0;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!result.long_names.empty())
                result.long_names += ' ';
            result.long_names += name;
        }
        result.alphabet += info.alphabet;
        result.base_counts[index++] = info.base_counts;
    }

    return result;
}

}  // namespace dorado
