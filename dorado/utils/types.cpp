#include "types.h"

#include "tensor_utils.h"

#include <htslib/sam.h>
#include <minimap.h>
#include <toml.hpp>

namespace dorado {

void BamDestructor::operator()(bam1_t* bam) { bam_destroy1(bam); }

const std::vector<int> RemoraUtils::BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

void ModBaseParams::parse(std::filesystem::path const& model_path, bool all_members) {
    auto config = toml::parse(model_path / "config.toml");
    const auto& params = toml::find(config, "modbases");
    motif = toml::find<std::string>(params, "motif");
    motif_offset = toml::find<int>(params, "motif_offset");

    mod_bases = toml::find<std::string>(params, "mod_bases");
    for (size_t i = 0; i < mod_bases.size(); ++i) {
        mod_long_names.push_back(
                toml::find<std::string>(params, "mod_long_names_" + std::to_string(i)));
    }

    if (!all_members) {
        return;
    }

    base_mod_count = mod_long_names.size();

    context_before = toml::find<int>(params, "chunk_context_0");
    context_after = toml::find<int>(params, "chunk_context_1");
    bases_before = toml::find<int>(params, "kmer_context_bases_0");
    bases_after = toml::find<int>(params, "kmer_context_bases_1");
    offset = toml::find<int>(params, "offset");

    if (config.contains("refinement")) {
        // these may not exist if we convert older models
        const auto& refinement_params = toml::find(config, "refinement");
        refine_do_rough_rescale =
                (toml::find<int>(refinement_params, "refine_do_rough_rescale") == 1);
        if (refine_do_rough_rescale) {
            refine_kmer_center_idx = toml::find<int>(refinement_params, "refine_kmer_center_idx");

            auto kmer_levels_tensor =
                    utils::load_tensors(model_path, {"refine_kmer_levels.tensor"})[0].contiguous();
            std::copy(kmer_levels_tensor.data_ptr<float>(),
                      kmer_levels_tensor.data_ptr<float>() + kmer_levels_tensor.numel(),
                      std::back_inserter(refine_kmer_levels));
            refine_kmer_len = static_cast<size_t>(
                    std::round(std::log(refine_kmer_levels.size()) / std::log(4)));
        }

    } else {
        // if the toml file doesn't contain any of the above parameters then
        // the model doesn't support rescaling so turn it off
        refine_do_rough_rescale = false;
    }
}

// Here mm_tbuf_t is used instead of mm_tbuf_s since minimap.h
// provides a typedef for mm_tbuf_s to mm_tbuf_t.
void MmTbufDestructor::operator()(mm_tbuf_t* tbuf) { mm_tbuf_destroy(tbuf); }

}  // namespace dorado
