#include "models.h"

#include "data_loader/ModelFinder.h"
#include "kits.h"
#include "metadata.h"
#include "model_downloader.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace fs = std::filesystem;

namespace dorado::models {

// Test if a ModelInfo matches optional criteria
bool model_info_is_similar(const ModelInfo& info,
                           const Chemistry chemistry,
                           const ModelVariantPair model,
                           const ModsVariantPair mods) {
    if (chemistry != Chemistry::NONE && chemistry != info.chemistry) {
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
    if (Chemistry::NONE == chemistry) {
        throw std::runtime_error("Cannot get model without chemistry");
    }

    if (mods.has_variant()) {
        const auto no_variant = ModsVariantPair{ModsVariant::NONE, mods.ver};
        const auto matches = find_models(models, chemistry, model, no_variant);
        if (matches.size() > 0) {
            spdlog::info("Found {} {} models without mods variant: {}", matches.size(), description,
                         to_string(mods.variant));
            for (auto m : matches) {
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
            for (auto m : matches) {
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
            for (auto m : matches) {
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
            for (auto m : matches) {
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

ModelInfo find_model(const std::vector<ModelInfo>& models,
                     const std::string& description,
                     const Chemistry& chemistry,
                     const ModelVariantPair& model,
                     const ModsVariantPair& mods,
                     bool suggestions) {
    if (Chemistry::NONE == chemistry) {
        throw std::runtime_error("Cannot get model without chemistry");
    }
    const auto matches = find_models(models, chemistry, model, mods);

    if (matches.empty()) {
        spdlog::error("Failed to get {} model", description);
        if (suggestions) {
            suggest_models(models, description, chemistry, model, mods);
        }
        throw std::runtime_error("No matches for " + format_msg(chemistry, model, mods));
    }

    // Get the only match or the latest model
    return matches.back();
}

std::vector<ModelInfo> find_models(const std::vector<ModelInfo>& models,
                                   const Chemistry& chemistry,
                                   const ModelVariantPair& model,
                                   const ModsVariantPair& mods) {
    if (Chemistry::NONE == chemistry) {
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

        // RNA004
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

        // RNA004 v3.0.1
        ModelInfo{
                "rna004_130bps_sup@v3.0.1_m6A_DRACH@v1",
                "356b3eed19916d83d59cbfd24bb9f33823d6f738891f3ac8fe77319ae5cbde7f",
                CC::RNA004_130BPS,
                ModelVariantPair{ModelVariant::SUP, VV::v3_0_1},
                ModsVariantPair{ModsVariant::M_m6A_DRACH, VV::v1_0_0},
        },
};

}  // namespace modified

const std::vector<ModelInfo>& simplex_models() { return simplex::models; }
const std::vector<ModelInfo>& stereo_models() { return stereo::models; }
const std::vector<ModelInfo>& modified_models() { return modified::models; }

namespace {
std::vector<std::string> unpack_names(const std::vector<ModelInfo>& infos) {
    std::vector<std::string> v;
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
    for (const auto& collection : {simplex::models, stereo::models, modified::models}) {
        for (const ModelInfo& model_info : collection) {
            if (model_info.name == model_name) {
                return true;
            }
        }
    }
    return false;
}

bool download_models(const std::string& target_directory, const std::string& selected_model) {
    if (selected_model != "all" && !is_valid_model(selected_model)) {
        spdlog::error("Selected model doesn't exist: {}", selected_model);
        return false;
    }

    ModelDownloader downloader(target_directory);

    bool success = true;
    auto download_model_set = [&](const ModelList& models) {
        for (const auto& model : models) {
            if (selected_model == "all" || selected_model == model.name) {
                if (!downloader.download(model.name, model)) {
                    success = false;
                }
            }
        }
    };

    download_model_set(simplex::models);
    download_model_set(stereo::models);
    download_model_set(modified::models);

    return success;
}

ModelInfo get_simplex_model_info(const std::string& model_name) {
    const auto simplex_model_infos = simplex_models();
    auto is_name_match = [&model_name](const ModelInfo& info) { return info.name == model_name; };
    std::vector<ModelInfo> matches;
    std::copy_if(simplex_model_infos.begin(), simplex_model_infos.end(),
                 std::back_inserter(matches), is_name_match);

    if (matches.empty()) {
        throw std::runtime_error("Could not find information on simplex model: " + model_name);
    } else if (matches.size() > 1) {
        throw std::logic_error("Found multiple simplex models with name: " + model_name);
    }
    return matches.back();
}

std::string get_modification_model(const std::filesystem::path& simplex_model_path,
                                   const std::string& modification) {
    std::string modification_model{""};

    if (!fs::exists(simplex_model_path)) {
        throw std::runtime_error{
                "Cannot find modification model for '" + modification +
                "' reason: simplex model doesn't exist at: " + simplex_model_path.u8string()};
    }

    auto model_dir = simplex_model_path.parent_path();
    auto simplex_name = simplex_model_path.filename().u8string();

    if (is_valid_model(simplex_name)) {
        std::string mods_prefix = simplex_name + "_" + modification + "@v";
        for (const auto& info : modified::models) {
            // There is an assumption that models with multiple versions
            // are named in a way that picking the last one after lexicographically
            // sorting them finds the latest version.
            if (utils::starts_with(info.name, mods_prefix)) {
                modification_model = info.name;
            }
        }
    } else {
        throw std::runtime_error{"Cannot find modification model for '" + modification +
                                 "' reason: unknown simplex model " + simplex_name};
    }

    if (modification_model.empty()) {
        throw std::runtime_error{"could not find matching modification model for " + simplex_name};
    }

    spdlog::debug("- matching modification model found: {}", modification_model);

    auto modification_path = model_dir / fs::path{modification_model};
    if (!fs::exists(modification_path)) {
        if (!download_models(model_dir.u8string(), modification_model)) {
            throw std::runtime_error("Failed to download model: " + modification_model);
        }
    }

    return modification_path.u8string();
}

SamplingRate get_sample_rate_by_model_name(const std::string& model_name) {
    const auto chemistries = chemistry_kits();
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

}  // namespace dorado::models
