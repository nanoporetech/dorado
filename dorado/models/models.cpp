#include "models.h"

#include "kits.h"
#include "metadata.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace dorado::models {

namespace {

// Test if a ModelInfo matches optional criteria
bool model_info_is_similar(const ModelInfo& info,
                           const Chemistry chemistry,
                           const ModelVariantPair model,
                           const ModsVariantPair mods) {
    if (chemistry != Chemistry::UNKNOWN && chemistry != info.chemistry) {
        return false;
    }

    const bool is_auto_variant_match = model.variant == ModelVariant::AUTO && info.simplex.is_auto;
    if (model.has_variant() && info.simplex.has_variant() &&
        info.simplex.variant != model.variant && !is_auto_variant_match) {
        spdlog::trace("Reject {} on model type: {}", info.name, to_string(model.variant));
        return false;
    }
    if (model.has_ver() && info.simplex.has_ver() && model.ver != info.simplex.ver) {
        spdlog::trace("Reject {} on versions: {}, {}", info.name, to_string(info.simplex.ver),
                      to_string(model.ver));
        return false;
    }
    if (mods.has_variant() && info.mods.has_variant() && mods.variant != info.mods.variant) {
        spdlog::trace("Reject {} on mods: {}, {}", info.name, to_string(info.mods.variant),
                      to_string(mods.variant));
        return false;
    }
    if (mods.has_ver() && info.mods.has_ver() && mods.ver != info.mods.ver) {
        spdlog::trace("Reject {} on mods versions: {}, {}", info.name, to_string(info.mods.ver),
                      to_string(mods.ver));
        return false;
    }
    return true;
}

// Sort model info by version in ascending order
void sort_versions(std::vector<ModelInfo>& infos) {
    auto cmp_version = [](const ModelInfo& a, const ModelInfo& b) {
        return std::tie(a.simplex.ver, a.mods.ver) < std::tie(b.simplex.ver, b.mods.ver);
    };
    std::sort(infos.begin(), infos.end(), cmp_version);
}

void suggest_models(const std::vector<ModelInfo>& models,
                    const std::string& description,
                    const Chemistry& chemistry,
                    const ModelVariantPair& model,
                    const ModsVariantPair& mods) {
    if (chemistry == Chemistry::UNKNOWN) {
        throw std::runtime_error("Cannot get model without chemistry");
    }

    if (mods.has_variant()) {
        const auto no_variant = ModsVariantPair{ModsVariant::NONE, mods.ver};
        const auto matches = find_models(models, chemistry, model, no_variant);
        if (matches.size() > 0) {
            spdlog::info("Found {} {} models without mods variant: {}", matches.size(), description,
                         to_string(mods.variant));
            for (const auto& m : matches) {
                spdlog::info("- {} - mods variant: {}", m.name, to_string(m.mods.variant));
            }
        }
    }
    if (mods.has_ver()) {
        const auto no_ver = ModsVariantPair{mods.variant, ModelVersion::NONE};
        const auto matches = find_models(models, chemistry, model, no_ver);
        if (matches.size() > 0) {
            spdlog::info("Found {} {} models without mods version: {}", matches.size(), description,
                         to_string(mods.ver));
            for (const auto& m : matches) {
                spdlog::info("- {} - mods version: {}", m.name, to_string(m.mods.ver));
            }
        }
    }

    if (mods.has_variant() || mods.has_ver()) {
        return;
    }

    if (model.has_variant()) {
        const auto no_variant = ModelVariantPair{ModelVariant::NONE, model.ver};
        const auto matches = find_models(models, chemistry, no_variant, mods);
        if (matches.size() > 0) {
            spdlog::info("Found {} {} models without model variant: {}", matches.size(),
                         description, to_string(model.variant));
            for (const auto& m : matches) {
                spdlog::info("- {} - model variant: {}", m.name, to_string(m.simplex.variant));
            }
        }
    }
    if (model.has_ver()) {
        const auto no_ver = ModelVariantPair{model.variant, ModelVersion::NONE};
        const auto matches = find_models(models, chemistry, no_ver, mods);
        if (matches.size() > 0) {
            spdlog::info("Found {} {} models without model version: {}", matches.size(),
                         description, to_string(model.ver));
            for (const auto& m : matches) {
                spdlog::info("- {} - model version: {}", m.name, to_string(m.simplex.ver));
            }
        }
    }
}

// Stringify the chemistry, model variants and mods variants for debugging / error messages
std::string format_msg(const Chemistry& chemistry,
                       const ModelVariantPair& model,
                       const ModsVariantPair& mods) {
    std::string s = "chemistry: " + to_string(chemistry);
    s += model.has_variant() ? ", model_variant: " + to_string(model.variant) : "";
    s += model.has_ver() ? ", version: " + to_string(model.ver) : "";
    s += mods.has_variant() ? ", mods_variant: " + to_string(mods.variant) : "";
    s += mods.has_ver() ? ", mods_version: " + to_string(mods.ver) : "";
    return s;
}

}  // namespace

ModelInfo find_model(const std::vector<ModelInfo>& models,
                     const std::string& description,
                     const Chemistry& chemistry,
                     const ModelVariantPair& model,
                     const ModsVariantPair& mods,
                     bool suggestions) {
    if (Chemistry::UNKNOWN == chemistry) {
        throw std::runtime_error("Cannot get model without chemistry");
    }
    const std::vector<ModelInfo> matches = find_models(models, chemistry, model, mods);

    if (matches.empty()) {
        spdlog::error("Failed to get {} model", description);
        if (suggestions) {
            suggest_models(models, description, chemistry, model, mods);
        }
        throw std::runtime_error("No matches for " + format_msg(chemistry, model, mods));
    }

    // Get the only match or the latest model as models are sorted in ascending version order
    const ModelInfo& selection = matches.back();
    if (matches.size() > 1) {
        spdlog::trace("Selected {} model: '{}' from {} matches.", description, selection.name,
                      matches.size());
    }
    return selection;
}

std::vector<ModelInfo> find_models(const std::vector<ModelInfo>& models,
                                   const Chemistry& chemistry,
                                   const ModelVariantPair& model,
                                   const ModsVariantPair& mods) {
    if (Chemistry::UNKNOWN == chemistry) {
        throw std::logic_error("Cannot get models without chemistry");
    }

    spdlog::trace("Searching for: {}", format_msg(chemistry, model, mods));

    auto is_match = [&chemistry, &model, &mods](const ModelInfo& info) {
        return model_info_is_similar(info, chemistry, model, mods);
    };

    std::vector<ModelInfo> matches;
    std::copy_if(models.begin(), models.end(), std::back_inserter(matches), is_match);
    sort_versions(matches);

    spdlog::trace("Found {} model matches:", matches.size());
    for (const auto& m : matches) {
        spdlog::trace("- {}", m.name);
    }
    return matches;
}

using CC = Chemistry;
using VV = ModelVersion;

// Serialised, released models
namespace simplex {

const std::vector<ModelInfo> models = {
        // v3.{3,4,6}
        ModelInfo{
                "dna_r9.4.1_e8_fast@v3.4",
                "879cbe2149d5eea524e8902a2d00b39c9b999b66ef40938f0cc37e7e0dc88aed",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::FAST, VV::v3_4_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_hac@v3.3",
                "6f74b6a90c70cdf984fed73798f5e5a8c17c9af3735ef49e83763143c8c67066",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::HAC, VV::v3_3_0, true},
        },
        ModelInfo{
                "dna_r9.4.1_e8_sup@v3.3",
                "5fc46541ad4d82b37778e87e65ef0a36b578b1d5b0c55832d80b056bee8703a4",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::SUP, VV::v3_3_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_sup@v3.6",
                "1db1377b516c158b5d2c39533ac62e8e334e70fcb71c0a4d29e7b3e13632aa73",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::SUP, VV::v3_6_0},
        },

        // v3.5.2 260bps
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
                "d2c9da317ca431da8adb9ecfc48f9b94eca31c18074062c0e2a8e2e19abc5c13",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v3_5_2},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
                "c3d4e017f4f7200e9622a55ded303c98a965868e209c08bb79cbbef98ffd552f",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v3_5_2},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v3.5.2",
                "51d30879dddfbf43f794ff8aa4b9cdf681d520cc62323842c2b287282326b4c5",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v3_5_2},
        },

        // v3.5.2 400bps
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
                "8d753ac1c30100a49928f7a722f18b14309b5d3417b5f12fd85200239058c36f",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v3_5_2},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
                "42e790cbb436b7298309d1e8eda7367e1de3b9c04c64ae4da8a28936ec5169f8",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v3_5_2},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v3.5.2",
                "4548b2e25655ce205f0e6fd851bc28a67d9dc13fea7d86efc00c26f227fa17ef",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v3_5_2},
        },

        // v4.0.0 260 bps
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
                "d79e19db5361590b44abb2b72395cc83fcca9f822eb3ce049c9675d5d87274dd",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v4_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
                "b523f6765859f61f48a2b65c061b099893f78206fe2e5d5689e4aebd6bf42adf",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v4_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v4.0.0",
                "7c3ab8a1dd89eab53ff122d7e76ff31acdb23a2be988eec9384c6a6715252e41",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v4_0_0},
        },

        // v4.0.0 400 bps
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
                "d826ccb67c483bdf27ad716c35667eb4335d9487a69e1ac87437c6aabd1f849e",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
                "b04a14de1645b1a0cf4273039309d19b66f7bea9d24bec1b71a58ca20c19d7a0",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
                "a6ca3afac78a25f0ec876f6ea507f42983c7da601d14314515c271551aef9b62",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_0_0},
        },

        // v4.1.0
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
                "5194c533fbdfbab9db590997e755501c65b609c5933943d3099844b83def95b5",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v4_1_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
                "0ba074e95a92e2c4912dbe2c227c5fa5a51e6900437623372b50d4e58f04b9fb",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v4_1_0, true},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v4.1.0",
                "c236b2a1c0a1c7e670f7bd07e6fd570f01a366538f7f038a76e9cafa62bbf7a4",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v4_1_0},
        },

        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
                "8a3d79e0163003591f01e273877cf936a344c8edc04439ee5bd65e0419d802f2",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_1_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
                "7da27dc97d45063f0911eac3f08c8171b810b287fd698a4e0c6b1734f02521bf",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_1_0, true},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",
                "47d8d7712341affd88253b5b018609d0caeb76fd929a8dbd94b35c1a2139e37d",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_1_0},
        },

        // v4.2.0
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
                "be62b912cdabb77b4a25ac9a83ee64ddd8b7fc75deaeb6975f5809c4a97d9c4b",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_2_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
                "859d12312cbf47a0c7a8461c26b507e6764590c477e1ea0605510022bbaa8347",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_2_0, true},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
                "87c8d044698e37dae1f9100dc4ed0567c6754dcffae446b5ac54a02c0efc401a",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
        },

        // v4.3.0
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.3.0",
                "3c38af7258071171976967eaff3a1713fba0ac09740388288a4a04a9eaf82075",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_3_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
                "83e2292dd577b094e41e6399a7fe0d45e29eee478bf8cfbccaff7f2e19180e95",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0, true},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0",
                "ee9515ca1c8aba1ad5c53f66ba9a560e5995cfd8eead76d208a877fc5dcf1901",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
        },

        // v5.0.0
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v5.0.0",
                "2625e2191a2e2a15cf5df11bcbab32d7f2712070c04eed6dfbd5b770dd7bbbc2",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v5_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                "b09d5c73d403c44d0d3e3cefb5ddc5bcee1f3d8cec4efd0872c16a32eba5ff41",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0, true},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0",
                "cbc7fa906cdfbbcaf5bf6d5dc0a20a63fa83e6ebefdd3aeda70724c47ca9f966",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
        },

        ModelInfo{
                "dna_r10.4.1_e8.2_apk_sup@v5.0.0",
                "863996f35e6857ff52140c0bfb5284d05efe40fcdab03faea7c97b9ab0747417",
                CC::DNA_R10_4_1_E8_2_APK_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
        },

        // RNA002
        ModelInfo{
                "rna002_70bps_fast@v3",
                "f8f533797e9bf8bbb03085568dc0b77c11932958aa2333902cf2752034707ee6",
                CC::RNA002_70BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v3_0_0},
        },
        ModelInfo{
                "rna002_70bps_hac@v3",
                "342b637efdf1a106107a1f2323613f3e4793b5003513b0ed85f6c76574800b52",
                CC::RNA002_70BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v3_0_0, true},
        },

        // RNA004 v3.0.1
        ModelInfo{
                "rna004_130bps_fast@v3.0.1",
                "2afa5de03f28162dd85b7be4a2dda108be7cc0a19062db7cb8460628aac462c0",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v3_0_1},
        },
        ModelInfo{
                "rna004_130bps_hac@v3.0.1",
                "0b57da141fe97a85d2cf7028c0d0b83c24be35451fd2f8bfb6070f82a1443ea0",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v3_0_1, true},
        },
        ModelInfo{
                "rna004_130bps_sup@v3.0.1",
                "dfe3749c3fbede7203db36ab51689c911d623700e6a24198d398ab927dd756a3",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v3_0_1},
        },
        // RNA v5.0.0
        ModelInfo{
                "rna004_130bps_fast@v5.0.0",
                "3b45ecedf2e20c56e15033402deb77f3c4e67df49aea8d7b76acdbb4029e8ea0",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v5_0_0},
        },
        ModelInfo{
                "rna004_130bps_hac@v5.0.0",
                "7ca33d19824e41e1c438a60d4d844c86d449be25ea59b8df23adee385a2d9f5d",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0, true},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.0.0",
                "c1fdcbc4eb75ec89ed43363f5360cb41541eb67ffa4729aabef843105bb07bb6",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
        },
        // RNA v5.1.0
        ModelInfo{
                "rna004_130bps_fast@v5.1.0",
                "c01353ac8362479ceedf607c41e5f238efd629725556d896161baa194b7354be",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v5_1_0},
        },
        ModelInfo{
                "rna004_130bps_hac@v5.1.0",
                "36ac8bdb2baaf32e697086962078f83a001a3ffe1461e358fabef15c08b15c5e",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_1_0, true},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.1.0",
                "ab7c5687f149901868898791b8d243c28e8345c9b61e3abce30d63e112ebc3b1",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_1_0},
        },
};

}  // namespace simplex

namespace stereo {

const std::vector<ModelInfo> models = {
        // Only 4kHz stereo model - matches all simplex models for this condition
        ModelInfo{
                "dna_r10.4.1_e8.2_4khz_stereo@v1.1",
                "d434525cbe1fd00adbd7f8a5f0e7f0bf09b77a9e67cd90f037c5ab52013e7974",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::NONE, VV::NONE},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_4khz_stereo@v1.1",
                "d434525cbe1fd00adbd7f8a5f0e7f0bf09b77a9e67cd90f037c5ab52013e7974",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::NONE, VV::NONE},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_5khz_stereo@v1.1",
                "6c16e3917a12ec297a6f5d1dc83c205fc0ac74282fffaf76b765995033e5f3d4",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::NONE, VV::v4_2_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_5khz_stereo@v1.2",
                "2631423b8843a82f69c8d4ab07fa554b7356a29f25c03424c26e7096d0e01418",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::NONE, VV::v4_3_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_5khz_stereo@v1.3",
                "6942f2d13b6509ae88eb3bd4f7a9f149b72edce817b800fe22fded6c7566dc10",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::NONE, VV::v5_0_0},
        },
};

}  // namespace stereo

namespace modified {

const std::vector<ModelInfo> models = {
        // v3.{3,4}
        ModelInfo{
                "dna_r9.4.1_e8_fast@v3.4_5mCG@v0.1",
                "dab18ae409c754ed164c0214b51d61a3b5126f3e5d043cee60da733db3e78b13",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::FAST, VV::v3_4_0},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v0_1_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_hac@v3.3_5mCG@v0.1",
                "349f6623dd43ac8a8ffe9b8e1a02dfae215ea0c1daf32120612dbaabb4f3f16d",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::HAC, VV::v3_3_0},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v0_1_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1",
                "7ee1893b2de195d387184757504aa5afd76d3feda1078dbc4098efe53acb348a",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::SUP, VV::v3_3_0},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v0_1_0},
        },

        ModelInfo{
                "dna_r9.4.1_e8_fast@v3.4_5mCG_5hmCG@v0",
                "d45f514c82f25e063ae9e9642d62cec24969b64e1b7b9dffb851b09be6e8f01b",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::FAST, VV::v3_4_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v0_0_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0",
                "4877da66a0ff6935033557a49f6dbc4676e9d7dba767927fec24b2deae3b681f",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::HAC, VV::v3_3_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v0_0_0},
        },
        ModelInfo{
                "dna_r9.4.1_e8_sup@v3.3_5mCG_5hmCG@v0",
                "7ef57e63f0977977033e3e7c090afca237e26fe3c94b950678346a1982f6116a",
                CC::DNA_R9_4_1_E8,
                ModelVariantPair{ModelVariant::SUP, VV::v3_3_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v0_0_0},
        },

        // v3.5.2
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2",
                "aa019589113e213f8a67c566874c60024584283de3d8a89ba0d0682c9ce8c2fe",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2",
                "bdbc238fbd9640454918d2429f909d9404e5897cc07b948a69462a4eec1838e0",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2",
                "0b528c5444c2ca4da7e265b846b24a13c784a34b64a7912fb50c14726abf9ae1",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },

        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2",
                "ac937da0224c481b6dbb0d1691ed117170ed9e7ff619aa7440123b88274871e8",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2",
                "50feb8da3f9b22c2f48d1c3e4aa495630b5f586c1516a74b6670092389bff56e",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2",
                "614604cb283598ba29242af68a74c5c882306922c4142c79ac2b3b5ebf3c2154",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v3_5_2},
                ModsVariantPair{ModsVariant::M_5mCG, VV::v2_0_0},
        },

        // v4.0.0
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v4.0.0_5mCG_5hmCG@v2",
                "b4178526838ed148c81c5189c013096768b58e9741c291fce71647613d93063a",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v4.0.0_5mCG_5hmCG@v2",
                "9447249b92febf5d856c247d39f2ce0655f9e2d3079c60b926ef1862e285951b",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2",
                "f41b7a8f53332bebedfd28fceba917e45c9a97aa2dbd21017999e3113cfb0dd3",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },

        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2",
                "91e242b5f58f2af843d8b7a975a31bcf8ff0a825bb0583783543c218811d427d",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2",
                "6926ae442b86f8484a95905f1c996c3672a76d499d00fcd0c0fbd6bd1f63fbb3",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2",
                "a7700b0e42779bff88ac02d6b5646b82dcfc65a418d83a8f6d8cca6e22e6cf97",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },

        // v4.1.0
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_fast@v4.1.0_5mCG_5hmCG@v2",
                "93c218d04c958f3559e18132977977ce4e8968e072bb003cab2fe05157c4ded0",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::FAST, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_hac@v4.1.0_5mCG_5hmCG@v2",
                "3178eb66d9e3480dae6e2b6929f8077d4e932820e7825c39b12bd8f381b9814a",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2",
                "d7a584f3c2abb6065014326201265ccce5657aec38eeca26d6d522a85b1e31cd",
                CC::DNA_R10_4_1_E8_2_260BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },

        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.1.0_5mCG_5hmCG@v2",
                "aa7af48a90752c15a4b5df5897035629b2657ea0fcc2c785de595c24c7f9e93f",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.1.0_5mCG_5hmCG@v2",
                "4c91b09d047d36dcb22e43b2fd85ef79e77b07009740ca5130a6a111aa60cacc",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.1.0_5mCG_5hmCG@v2",
                "73d20629445d21a27dc18a2622063a5916cb04938aa6f12c97ae6b77a883a832",
                CC::DNA_R10_4_1_E8_2_400BPS_4KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_1_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },

        // v4.2.0
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2",
                "a01761e709fd6c114b09ffc7100efb52c37faa38a3f8b281edf405904f04fefa",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::FAST, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_5mCG_5hmCG@v2",
                "2112aa355757906bfb815bf178fee260ad90cd353781ee45c121024c5caa7c6b",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v2",
                "6b3604799d85e81d06c97181af093b30483cec9ad02f54a631eca5806f7848ef",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v3.1",
                "5f8016f1b47e3c31825233e1eac8b7074bd61705cb5dfeca9e588d5077b18b66",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v3_1_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2",
                "61ecdba6292637942bc9f143180054084f268d4f8a7e1c7a454413519d5458a7",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mC, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2",
                "0f268e2af4db1023217ee01f2e2e23d47865fde5a5944d915fdb7572d92c0cb5",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v3",
                "903fb89e7c8929a3a66abf60eb6f1e1a7ab7b7e4a0c40f646dc0b13d5588174c",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v3_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC_5hmC@v1",
                "28d82762af14e18dd36fb1d9f044b1df96fead8183d3d1ef47a5e92048a2be27",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v1_0_0},
        },

        // V4.3.0
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_5mC_5hmC@v1",
                "03523262df93d75fc26e10fb05e3cd6459b233ec7545859c0f7fd3d4665768c1",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mC_5hmC@v1",
                "11ccf924cd0c28aff7e99e8f2acc88cd45f39e03496c61848f2ec0ede35ee547",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_6mA@v1",
                "68a5395f2773f755d2b25df89c3aa32a759e8909d1549967665f902b82588891",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_6mA@v1",
                "a1703971ec0b35af178180d1f23908f8587888c3bc3b727b230e6cd3eb575422",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_6mA@v2",
                "7b8e2887ba113832063555a0bc4df0e27ae2d905dbf7b65d05d7f91cf07df670",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_6mA@v2",
                "643891d0cafcb07e6f985b17ed2fe3e033feff4db9c4c3053faa5e3281b4b5b4",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_5mCG_5hmCG@v1",
                "49b1f6e1ae353bf0991c0001a47bdb9d2c01e097b60229ec6f576ff1d02bf604",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_5mCG_5hmCG@v1",
                "14af8002f5dfdce0c19e17a72620a29e58a988008e0aa9f8172e2fa2b2fedb5d",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v1_0_0},
        },
        // DNA V5.0.0
        // 4mC+5mC all-context HAC and SUP
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_4mC_5mC@v1",
                "e3f8b05ac202a9023b800cd1619a01ebf9ffd0ef0e731722a425f258f0b205cd",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_4mC_5mC@v1",
                "dfbef77e122d69805dc59cb23afd238af597369e289c1ee1f5463c5221e6d61a",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_4mC_5mC@v2",
                "d7c4ee43e954b081a0179e5236245a62094fcecb1454de2b3901f2b10d8807d7",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_4mC_5mC@v3",
                "4e78cc53d78f09bd50de59365633d8045e2b5cb8d1ed5854f997f8ea7cb61c0f",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v3_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_4mC_5mC@v2",
                "eb971340e111ebfdb27bd2b70390c5f0252ba91f5ac92eea0dbd59524bac68f7",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_4mC_5mC@v3",
                "55402c4cb087cdde5d0cfa573d0207b160a68d1ccd9b32c525d65bb6a503269a",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_4mC_5mC, VV::v3_0_0},
        },
        // 5mC+5hmC all-context HAC and SUP
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v1",
                "77af422ac0ef9f9383c2f396d06aa225b618507124a797250dc2e491d4fd634b",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v1",
                "f4b3e8ebe49b2f2f02be958e073fbff130995e55d158ec93c710824373fcfdd3",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v2",
                "8bde4f0fd27a2e2fbf98942a6e1cc1d4547c6678a69940a2152a6a5cdb98cc3c",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v3",
                "82e059428b82395468fdd6e90150517b22b3651c189fbd2f67a3b07ed64d1d03",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v3_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v2.0.1",
                "757dabc280e25f1c442fcfeb3e1f4d44a2d445e0ea89bb30c15e4757879111be",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v2_0_1},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v3",
                "9384a143fa08b946ff5f43a9f5a85e395c36df3b4895e418674099f8554956da",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mC_5hmC, VV::v3_0_0},
        },
        // 5mC+5hmC CG-context HAC and SUP
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v1",
                "0311a4c1573cfd02e9075e4c23137cd17c59f7b235805daa6cd769f101facc33",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v1",
                "fb6a25b50259e305e493d2b202e09973e08ae6347d5fd7554aa96a4c2264d82e",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v2",
                "5c2452e4ccd443e7f6549afb6ac732b03b90480801e2a29850e5616185cb6d5b",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v3",
                "df061855267edc5abdc9fa4c3add54390f08f2f80274c11ca90b74483be84ab7",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v3_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v2.0.1",
                "c8ebafd13008a919232cd45514e07ea929509a5e20254c73b9eff2cd0e5a4786",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v2_0_1},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v3",
                "0f6cc5d165fd25acb4227d2d791c017a59a21cc30d2a92b8c2925f84138974ee",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_5mCG_5hmCG, VV::v3_0_0},
        },
        // 6mA all-context HAC and SUP
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v1",
                "db064e9097f11a7f4eacffea4c85491b15176756c536cb92ed6b13dd8fa47305",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v1",
                "01755ea73e722f6f7fe4e0bee1060c9cc1c62f0f9e8a5ebd762385691d1564cf",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v1_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v2",
                "919aaf7fdfbf50a1fe20124e07014fa2b38cc10f3dadb27c56b415309147eee9",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v3",
                "7df772281e2d73e72c91b0a3b53e2487d4c4b62ad34dfe8d1296650a79b1920d",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v3_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v2",
                "fc1d247475162d4f782d66bb3cd6f19c76e5589a8e064f738de4896f940568b3",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v2_0_0},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_6mA@v3",
                "d7a6f9a52218a6996d31127d4c0513b96358721ac3ddaa4b008f9d7513d53473",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_6mA, VV::v3_0_0},
        },
        // RNA004 v3.0.1
        // m6A - DRACH
        ModelInfo{
                "rna004_130bps_sup@v3.0.1_m6A_DRACH@v1",
                "356b3eed19916d83d59cbfd24bb9f33823d6f738891f3ac8fe77319ae5cbde7f",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v3_0_1},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
        // RNA004 v5.0.0
        // m6A - all context
        ModelInfo{
                "rna004_130bps_hac@v5.0.0_m6A@v1",
                "34ae823429d00e38f9fe4a21e6ae0de70e6362e7f030da981752248c9d0f7f46",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_m6A, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.0.0_m6A@v1",
                "83d0e0432f6021a560f7b9ee1eea9ed54dd734ed80cd3bfefa98a20eae81a319",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_m6A, VV::v1_0_0},
        },
        // m6A - DRACH
        ModelInfo{
                "rna004_130bps_hac@v5.0.0_m6A_DRACH@v1",
                "b140acbfc04bb24080b39cc81d71016895dc74454c7cb630629b93ec60e315c9",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.0.0_m6A_DRACH@v1",
                "62dd2d9e225fa9638258bd33063fa930c4179b13878064547d5be7b33d478b23",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
        // pseU - all context
        ModelInfo{
                "rna004_130bps_hac@v5.0.0_pseU@v1",
                "bdb219bae25676f4584293a5494be1813360ac1829bc694995e4eaf105ebac79",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_pseU, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.0.0_pseU@v1",
                "d9ed391ce75eb47d10638dfaaea6081fa6bfb573feb4a4c7de040549559c2442",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{ModsVariant::M_pseU, VV::v1_0_0},
        },

        // RNA004 v5.1.0
        // m5C - all context
        ModelInfo{
                "rna004_130bps_hac@v5.1.0_m5C@v1",
                "d9c142ba65c15cebaf42ea44a3e5731bc3d59f89a2b07e55701f7152bde2937e",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_m5C, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.1.0_m5C@v1",
                "073a9a66a613f61fca83447816c4fd95ce608c854b54540e2a9f82b4c1498a3a",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_m5C, VV::v1_0_0},
        },
        // inosine_m6A - all context
        ModelInfo{
                "rna004_130bps_hac@v5.1.0_inosine_m6A@v1",
                "e709c9ce7e256f8d2bb259a0ab22d2bddc60c61834d3a020e2c8fc5721c5d548",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_inosine_m6A, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.1.0_inosine_m6A@v1",
                "8bcbd48f9f01eb624a8fdcb928c204b915ed002c1ddc600dfa3c2be16879b7df",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_inosine_m6A, VV::v1_0_0},
        },
        // m6A - DRACH
        ModelInfo{
                "rna004_130bps_hac@v5.1.0_m6A_DRACH@v1",
                "911ba609b657f8e24fe44519a965d0d9bac91f35e7026c8ee1614492bf7ce3f9",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.1.0_m6A_DRACH@v1",
                "ec616e5d725860e1686c17d70c8f135c6e0e66f6c3e7e28a6cdefe19cae2e91f",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
        // pseU - all context
        ModelInfo{
                "rna004_130bps_hac@v5.1.0_pseU@v1",
                "5d7c3cf12736baaba987c2ca899abd89193e859edfc7b9aad82a00e4bbc2e6bd",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::HAC, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_pseU, VV::v1_0_0},
        },
        ModelInfo{
                "rna004_130bps_sup@v5.1.0_pseU@v1",
                "02049be4f690cdf4a1200f6077b657c43587d1be2816fab01bb3f02f06e2cb7c",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v5_1_0},
                ModsVariantPair{ModsVariant::M_pseU, VV::v1_0_0},
        }};

}  // namespace modified

namespace correction {

const std::vector<ModelInfo> models = {
        ModelInfo{
                "herro-v1",
                "c7077840f84b469f2c0fd2ae44649fa5f7fa45132540fb54536792c8e22dab9a",
                CC::UNKNOWN,
                ModelVariantPair{},
                ModsVariantPair{},
        },
};

}  // namespace correction

namespace polisher {

const std::vector<ModelInfo> models = {
        // Read-level models.
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl",
                "d343b4394b904d219257ad188c82ece63b935f15d78f09f551e591b2275da4b9",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v5.0.0_polish_rl_mv",
                "928d9bcf3d68162eff479ada5839c5df3faa0ad393658729511aedffe65f089c",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v5_0_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_polish_rl",
                "6d8c5a8ce45311c25f824453d0af997fbe2f63a5f734fdb4d884d285ddafec33",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v5.0.0_polish_rl_mv",
                "0e0cb175aa41636de835d2abb5330b91fed14a00f811804edf983bc086cf477a",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v5_0_0},
                ModsVariantPair{},
        },

        // Bacterial models.
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_polish_bacterial_methylation_v5.0.0",
                "56e3763638677adb32de783e648dd721aa1ec04504d0328db70be208822aef6e",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{},
                ModsVariantPair{},
        },

        // Legacy models, counts based.
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_polish",
                "8092073ee021ac94e94a18000c97ac0e26dbf37f0cb16a706d583c2a374de33b",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_2_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_polish",
                "91ff09a711162b116c898f39831228d53e5e11981f6a16e08a7ef9575b911dc6",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_2_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_hac@v4.3.0_polish",
                "9bc24d9cd8c2247e472c3e5cbd6248cd00cf9547537bd3e2e6cac6334005700a",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::HAC, VV::v4_3_0},
                ModsVariantPair{},
        },
        ModelInfo{
                "dna_r10.4.1_e8.2_400bps_sup@v4.3.0_polish",
                "378f2e407f14bb2c4ec77e73b1b8ea86d3f4c47007d8ffb21f82fe6e89df7631",
                CC::DNA_R10_4_1_E8_2_400BPS_5KHZ,
                ModelVariantPair{ModelVariant::SUP, VV::v4_3_0},
                ModsVariantPair{},
        },
};

}  // namespace polisher

const std::vector<ModelInfo>& simplex_models() { return simplex::models; }
const std::vector<ModelInfo>& stereo_models() { return stereo::models; }
const std::vector<ModelInfo>& modified_models() { return modified::models; }
const std::vector<ModelInfo>& correction_models() { return correction::models; }
const std::vector<ModelInfo>& polish_models() { return polisher::models; }

namespace {
std::vector<std::string> unpack_names(const std::vector<ModelInfo>& infos) {
    std::vector<std::string> v;
    v.reserve(infos.size());
    for (const auto& item : infos) {
        v.push_back(std::string(item.name));
    }
    return v;
}
}  // namespace

std::vector<std::string> simplex_model_names() { return unpack_names(simplex_models()); };
std::vector<std::string> stereo_model_names() { return unpack_names(stereo_models()); };
std::vector<std::string> modified_model_names() { return unpack_names(modified_models()); };

std::vector<std::string> modified_model_variants() {
    std::vector<std::string> variants;
    for (const auto& kv : mods_variants_map()) {
        variants.push_back(kv.first);
    }
    return variants;
};

// Returns true if model_name matches any configured model
bool is_valid_model(const std::string& model_name) {
    for (const auto& collection : {simplex::models, stereo::models, modified::models,
                                   correction::models, polisher::models}) {
        for (const ModelInfo& model_info : collection) {
            if (model_info.name == model_name) {
                return true;
            }
        }
    }
    return false;
}

ModelInfo get_modification_model(const std::filesystem::path& simplex_model_path,
                                 const std::string& modification) {
    if (!fs::exists(simplex_model_path)) {
        throw std::runtime_error{
                "Cannot find modification model for '" + modification +
                "' reason: simplex model doesn't exist at: " + simplex_model_path.u8string()};
    }

    ModelInfo modification_model;
    bool model_found = false;
    auto simplex_name = simplex_model_path.filename().u8string();

    if (is_valid_model(simplex_name)) {
        std::string mods_prefix = simplex_name + "_" + modification + "@v";
        for (const auto& info : modified_models()) {
            // There is an assumption that models with multiple versions
            // are named in a way that picking the last one after lexicographically
            // sorting them finds the latest version.
            if (utils::starts_with(info.name, mods_prefix)) {
                modification_model = info;
                model_found = true;
            }
        }
    } else {
        throw std::runtime_error{"Cannot find modification model for '" + modification +
                                 "' reason: unknown simplex model '" + simplex_name + "'"};
    }

    if (!model_found) {
        throw std::runtime_error{"Cannot find modification model for '" + modification +
                                 "' matching simplex model: '" + simplex_name + "'"};
    }

    spdlog::debug("- matching modification model found: {}", modification_model.name);

    return modification_model;
}

ModelInfo get_simplex_model_info(const std::string& model_name) {
    const auto& simplex_model_infos = simplex_models();
    auto is_name_match = [&model_name](const ModelInfo& info) { return info.name == model_name; };
    std::vector<ModelInfo> matches;
    std::copy_if(simplex_model_infos.begin(), simplex_model_infos.end(),
                 std::back_inserter(matches), is_name_match);

    if (matches.empty()) {
        throw std::runtime_error("Could not find simplex model information from: " + model_name);
    } else if (matches.size() > 1) {
        throw std::logic_error("Found multiple simplex models with name: " + model_name);
    }
    return matches.back();
}

ModelInfo get_model_info(const std::string& model_name) {
    const auto& simplex_model_infos = simplex_models();
    const auto& mods_model_infos = modified_models();
    const auto& stereo_model_infos = stereo_models();
    const auto& correction_model_infos = correction_models();
    const auto& polish_model_infos = polish_models();

    auto is_name_match = [&model_name](const ModelInfo& info) { return info.name == model_name; };
    std::vector<ModelInfo> matches;
    std::copy_if(simplex_model_infos.begin(), simplex_model_infos.end(),
                 std::back_inserter(matches), is_name_match);
    std::copy_if(mods_model_infos.begin(), mods_model_infos.end(), std::back_inserter(matches),
                 is_name_match);
    std::copy_if(stereo_model_infos.begin(), stereo_model_infos.end(), std::back_inserter(matches),
                 is_name_match);
    std::copy_if(correction_model_infos.begin(), correction_model_infos.end(),
                 std::back_inserter(matches), is_name_match);
    std::copy_if(polish_model_infos.begin(), polish_model_infos.end(), std::back_inserter(matches),
                 is_name_match);

    if (matches.empty()) {
        throw std::runtime_error("Could not find information on model: " + model_name);
    } else if (matches.size() > 1) {
        throw std::logic_error("Found multiple models with name: " + model_name);
    }
    return matches.back();
}

SamplingRate get_sample_rate_by_model_name(const std::string& model_name) {
    const auto& chemistries = chemistry_kits();
    const ModelInfo model_info = get_simplex_model_info(model_name);
    auto iter = chemistries.find(model_info.chemistry);
    if (iter != chemistries.end()) {
        return iter->second.sampling_rate;
    } else {
        // This can only happen if a model_info.chemistry not in chemistries which should be impossible.
        throw std::logic_error("Couldn't find chemistry: " + to_string(model_info.chemistry));
    }
}

std::string extract_model_name_from_path(const std::filesystem::path& model_path) {
    return std::filesystem::canonical(model_path).filename().string();
}

std::string extract_model_names_from_paths(const std::vector<std::filesystem::path>& model_paths) {
    std::string model_names;
    for (const auto& model_path : model_paths) {
        if (!model_names.empty()) {
            model_names += ",";
        }
        model_names += models::extract_model_name_from_path(model_path);
    }
    return model_names;
}

namespace {

bool model_exists_in_folder(const std::string& name,
                            const std::filesystem::path& model_download_folder) {
    if (model_download_folder.empty()) {
        return true;
    }
    auto model_path = model_download_folder / name;
    return std::filesystem::exists(model_path) && std::filesystem::is_directory(model_path);
}

}  // namespace

std::string get_supported_model_info(const std::filesystem::path& model_download_folder) {
    std::string result = "{\n";

    const auto& canonical_base_map = mods_canonical_base_map();
    bool chemistry_emitted = false;
    for (const auto& variant : chemistry_kits()) {
        const auto& chemistry = variant.first;
        const auto& chemistry_kit_info = variant.second;

        if (chemistry == Chemistry::UNKNOWN) {
            continue;
        }

        // Check if there are any available canonical models for this chemistry - if not, we don't emit it.
        bool simplex_model_found = false;
        for (const auto& simplex_model : simplex_models()) {
            if (simplex_model.chemistry == variant.first &&
                model_exists_in_folder(simplex_model.name, model_download_folder)) {
                simplex_model_found = true;
                break;  // We can stop
            }
        }
        if (!simplex_model_found) {
            continue;
        }
        chemistry_emitted = true;

        // Chemistry name
        result += "\"" + chemistry_kit_info.name + "\":{\n";

        const auto& flowcell_code_map = flowcell_codes();
        const auto& kit_code_map = kit_codes();

        // Add some info on compatible kits and flowcells
        const auto& sample_type_name = get_sample_type_info(chemistry_kit_info.sample_type).name;
        result += "  \"sample_type\": \"" + sample_type_name + "\",\n";
        result +=
                "  \"sampling_rate\": " + std::to_string(chemistry_kit_info.sampling_rate) + ",\n";

        // Get the union of all supported flowcells and kits
        std::set<Flowcell> supported_flowcell_codes;
        std::set<KitCode> supported_kit_codes;
        for (const auto& kitset : chemistry_kit_info.kit_sets) {
            for (const auto& flowcell : kitset.first) {
                supported_flowcell_codes.insert(flowcell);
            }
            for (const auto& kit : kitset.second) {
                supported_kit_codes.insert(kit);
            }
        }

        std::string flowcells, kits;
        for (const auto& flowcell : supported_flowcell_codes) {
            flowcells += "    \"" + flowcell_code_map.at(flowcell).name + "\",\n";
        }
        flowcells = flowcells.substr(0, flowcells.length() - 2);  // trim last ",\n"
        for (const auto& kit : supported_kit_codes) {
            kits += "    \"" + kit_code_map.at(kit).name + "\",\n";
        }
        kits = kits.substr(0, kits.length() - 2);  // trim last ",\n"
        result += "  \"flowcells\":[\n" + flowcells + "\n  ],\n";
        result += "  \"kits\":[\n" + kits + "\n  ],\n";

        result += "  \"simplex_models\":{\n";
        for (const auto& simplex_model : simplex_models()) {
            if (simplex_model.chemistry == variant.first &&
                model_exists_in_folder(simplex_model.name, model_download_folder)) {
                result += "    \"" + simplex_model.name + "\":{\n";

                result += "      \"variant\": \"" + to_string(simplex_model.simplex.variant) + "\"";

                // If there is a newer model for this condition, add the outdated flag.
                const auto simplex_matches = find_models(
                        simplex_models(), variant.first,
                        {simplex_model.simplex.variant, ModelVersion::NONE}, ModsVariantPair());
                if (simplex_matches.back().name != simplex_model.name) {
                    result += ",\n      \"outdated\": true";
                }

                // Dump out all the mod models that are compatible with this simplex model
                const auto mod_matches = find_models(modified_models(), variant.first,
                                                     simplex_model.simplex, ModsVariantPair());
                bool mod_models_available = false;
                for (const auto& mod_model : mod_matches) {
                    if (model_exists_in_folder(mod_model.name, model_download_folder)) {
                        mod_models_available = true;
                        break;
                    }
                }

                if (mod_models_available) {
                    result += ",\n      \"modified_models\":{\n";
                    for (const auto& mod_model : mod_matches) {
                        if (model_exists_in_folder(mod_model.name, model_download_folder)) {
                            result += "        \"" + mod_model.name + "\":{\n";
                            result += "          \"canonical_base\": \"" +
                                      canonical_base_map.at(mod_model.mods.variant) + "\",\n";
                            result += "          \"context\": \"" +
                                      get_mods_context(mod_model.mods.variant) + "\",\n";
                            result += "          \"variant\": \"" +
                                      to_string(mod_model.mods.variant) + "\"";
                            // If there is a newer model for this condition, add the outdated flag.
                            const auto mod_type_matches = find_models(
                                    modified_models(), variant.first, simplex_model.simplex,
                                    {mod_model.mods.variant, ModelVersion::NONE});
                            if (mod_type_matches.back().name != mod_model.name) {
                                result += ",\n          \"outdated\": true";
                            }
                            result += "\n        },\n";
                        }
                    }
                    result = result.substr(0, result.length() - 2);  // trim last ",\n"
                    result += "\n      }";
                }

                const auto stereo_matches = find_models(stereo_models(), variant.first,
                                                        simplex_model.simplex, ModsVariantPair());
                bool stereo_models_available = false;
                for (const auto& stereo_model : stereo_matches) {
                    if (model_exists_in_folder(stereo_model.name, model_download_folder)) {
                        stereo_models_available = true;
                        break;
                    }
                }

                if (stereo_models_available) {
                    result += ",\n      \"stereo_models\":{\n";
                    for (const auto& stereo_model : stereo_matches) {
                        if (model_exists_in_folder(stereo_model.name, model_download_folder)) {
                            result += "        \"" + stereo_model.name + "\":{\n";
                            // If there is a newer model for this condition, add the outdated flag.
                            const auto duplex_type_matches = find_models(
                                    stereo_models(), variant.first, simplex_model.simplex,
                                    {stereo_model.mods.variant, ModelVersion::NONE});
                            if (duplex_type_matches.back().name != stereo_model.name) {
                                result += ",\n            \"outdated\": true";
                            }
                            result += "\n        },\n";
                        }
                    }
                    result = result.substr(0, result.length() - 2);  // trim last ",\n"
                    result += "\n      }";
                }

                // Polishing models.
                {
                    const auto polish_matches =
                            find_models(polish_models(), variant.first, simplex_model.simplex,
                                        ModsVariantPair());
                    bool polish_models_available = false;
                    for (const auto& polish_model : polish_matches) {
                        if (model_exists_in_folder(polish_model.name, model_download_folder)) {
                            polish_models_available = true;
                            break;
                        }
                    }
                    if (polish_models_available) {
                        result += ",\n      \"polish_models\":{\n";
                        for (const auto& polish_model : polish_matches) {
                            if (model_exists_in_folder(polish_model.name, model_download_folder)) {
                                result += "        \"" + polish_model.name + "\":{\n";
                                result += "\n        },\n";
                            }
                        }
                        result = result.substr(0, result.length() - 2);  // trim last ",\n"
                        result += "\n      }";
                    }
                }

                result += "\n    },\n";
            }
        }
        result = result.substr(0, result.length() - 2);  // trim last ",\n"
        result += "\n  }\n},\n";
    }
    if (chemistry_emitted) {
        result = result.substr(0, result.length() - 2);  // trim last ",\n"
    }

    result += "\n}\n";

    return result;
}

}  // namespace dorado::models
