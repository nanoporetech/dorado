#include "ModBaseModelConfig.h"

#include "torch_utils/tensor_utils.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <toml.hpp>

namespace dorado::modbase {

ModBaseModelConfig load_modbase_model_config(const std::filesystem::path& model_path) {
    ModBaseModelConfig config;
    auto config_toml = toml::parse(model_path / "config.toml");
    const auto& params = toml::find(config_toml, "modbases");
    config.motif = toml::find<std::string>(params, "motif");
    config.motif_offset = toml::find<int>(params, "motif_offset");

    const std::string canonical_bases = "ACGT";
    std::string motif_base = config.motif.substr(config.motif_offset, 1);
    if (canonical_bases.find(motif_base) == std::string::npos) {
        throw std::runtime_error("Invalid base for modification: " + motif_base);
    }

    const auto& mod_bases = toml::find(params, "mod_bases");
    if (mod_bases.is_string()) {
        auto mod_base_string = mod_bases.as_string().str;
        for (const auto& mod_base : mod_base_string) {
            config.mod_bases.push_back(std::string(1, mod_base));
        }
    } else {
        auto mod_base_array = mod_bases.as_array();
        for (const auto& mod_base : mod_base_array) {
            assert(mod_base.is_string());
            config.mod_bases.push_back(mod_base.as_string().str);
        }
    }

    for (const auto& mod_base : config.mod_bases) {
        if (!utils::validate_bam_tag_code(mod_base)) {
            throw std::runtime_error("Invalid modified base code: " + mod_base);
        }
    }

    for (size_t i = 0; i < config.mod_bases.size(); ++i) {
        config.mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    config.base_mod_count = config.mod_bases.size();

    config.context_before = toml::find<int>(params, "chunk_context_0");
    config.context_after = toml::find<int>(params, "chunk_context_1");
    config.bases_before = toml::find<int>(params, "kmer_context_bases_0");
    config.bases_after = toml::find<int>(params, "kmer_context_bases_1");
    config.offset = toml::find<int>(params, "offset");

    if (params.contains("reverse_signal")) {
        config.reverse_signal = toml::find<bool>(params, "reverse_signal");
    } else {
        config.reverse_signal = false;
    }

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
        const std::vector<std::reference_wrapper<const ModBaseModelConfig>>& base_mod_params) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::vector<std::string> alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    const std::string allowed_bases = "ACGT";
    std::array<ModelInfo, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet.emplace_back(1, allowed_bases[b]);
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = params_ref.get();
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            throw std::runtime_error("Invalid base in remora model metadata.");
        }
        auto& map_entry = model_info[utils::BaseInfo::BASE_IDS[base]];
        map_entry.long_names = params.mod_long_names;
        map_entry.alphabet.insert(map_entry.alphabet.end(), params.mod_bases.begin(),
                                  params.mod_bases.end());
        map_entry.base_counts = params.base_mod_count + 1;
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
    std::string err_msg = "";
    for (size_t i = 0; i < modbase_models.size(); i++) {
        auto ref_model = load_modbase_model_config(modbase_models[i]);
        const auto& ref_motif = ref_model.motif[ref_model.motif_offset];
        for (size_t j = i + 1; j < modbase_models.size(); j++) {
            auto query_model = load_modbase_model_config(modbase_models[j]);
            const auto& query_motif = query_model.motif[query_model.motif_offset];

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
