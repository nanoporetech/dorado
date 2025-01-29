#include "models/metadata.h"

#include <catch2/catch_test_macros.hpp>

#include <set>
#include <stdexcept>

#define TEST_TAG "[ModelMetadata]"

using namespace dorado::models;

CATCH_TEST_CASE(TEST_TAG "  ModelVariant enumeration", TEST_TAG) {
    const auto& codes = model_variants_map();

    CATCH_SECTION("Only expected ModelVariant exist") {
        CATCH_CHECK(codes.at("auto") == ModelVariant::AUTO);
        CATCH_CHECK(codes.at("fast") == ModelVariant::FAST);
        CATCH_CHECK(codes.at("hac") == ModelVariant::HAC);
        CATCH_CHECK(codes.at("sup") == ModelVariant::SUP);
        CATCH_CHECK(codes.size() == static_cast<size_t>(ModelVariant::NONE));
    }

    CATCH_SECTION("ModelVariant get_model_code") {
        CATCH_CHECK(get_model_variant("auto") == ModelVariant::AUTO);
        CATCH_CHECK(get_model_variant("fast") == ModelVariant::FAST);
        CATCH_CHECK(get_model_variant("hac") == ModelVariant::HAC);
        CATCH_CHECK(get_model_variant("sup") == ModelVariant::SUP);

        for (const auto& it : {"", "foo", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2"}) {
            CATCH_CHECK(get_model_variant(it) == ModelVariant::NONE);
        }
    }

    CATCH_SECTION("ModelVariant to_string") {
        CATCH_CHECK(to_string(ModelVariant::AUTO) == "auto");
        CATCH_CHECK(to_string(ModelVariant::FAST) == "fast");
        CATCH_CHECK(to_string(ModelVariant::HAC) == "hac");
        CATCH_CHECK(to_string(ModelVariant::SUP) == "sup");
        CATCH_CHECK_THROWS_AS(to_string(ModelVariant::NONE), std::logic_error);
    }

    CATCH_SECTION("ModelVariant self consistent") {
        for (const auto& code : codes) {
            if (code.second != ModelVariant::NONE) {
                CATCH_CHECK(code.second == get_model_variant(to_string(code.second)));
            }
        }
    }
}

CATCH_TEST_CASE(TEST_TAG "  ModsVariant enumeration", TEST_TAG) {
    const auto& mods = mods_variants_map();

    CATCH_SECTION("Only expected ModsVariant exist") {
        CATCH_CHECK(mods.at("4mC_5mC") == ModsVariant::M_4mC_5mC);
        CATCH_CHECK(mods.at("5mC_5hmC") == ModsVariant::M_5mC_5hmC);
        CATCH_CHECK(mods.at("5mCG") == ModsVariant::M_5mCG);
        CATCH_CHECK(mods.at("5mCG_5hmCG") == ModsVariant::M_5mCG_5hmCG);
        CATCH_CHECK(mods.at("5mC") == ModsVariant::M_5mC);
        CATCH_CHECK(mods.at("m5C") == ModsVariant::M_m5C);
        CATCH_CHECK(mods.at("6mA") == ModsVariant::M_6mA);
        CATCH_CHECK(mods.at("m6A") == ModsVariant::M_m6A);
        CATCH_CHECK(mods.at("m6A_DRACH") == ModsVariant::M_m6A_DRACH);
        CATCH_CHECK(mods.at("inosine_m6A") == ModsVariant::M_inosine_m6A);
        CATCH_CHECK(mods.at("pseU") == ModsVariant::M_pseU);
        CATCH_CHECK(mods.size() == static_cast<size_t>(ModsVariant::NONE));
    }

    CATCH_SECTION("ModsVariant get_mods_code") {
        CATCH_CHECK(get_mods_variant("4mC_5mC") == ModsVariant::M_4mC_5mC);
        CATCH_CHECK(get_mods_variant("5mC_5hmC") == ModsVariant::M_5mC_5hmC);
        CATCH_CHECK(get_mods_variant("5mCG") == ModsVariant::M_5mCG);
        CATCH_CHECK(get_mods_variant("5mCG_5hmCG") == ModsVariant::M_5mCG_5hmCG);
        CATCH_CHECK(get_mods_variant("5mC") == ModsVariant::M_5mC);
        CATCH_CHECK(get_mods_variant("m5C") == ModsVariant::M_m5C);
        CATCH_CHECK(get_mods_variant("6mA") == ModsVariant::M_6mA);
        CATCH_CHECK(get_mods_variant("m6A") == ModsVariant::M_m6A);
        CATCH_CHECK(get_mods_variant("m6A_DRACH") == ModsVariant::M_m6A_DRACH);
        CATCH_CHECK(get_mods_variant("inosine_m6A") == ModsVariant::M_inosine_m6A);
        CATCH_CHECK(get_mods_variant("pseU") == ModsVariant::M_pseU);
        for (const auto& it : {"", "foo", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2"}) {
            CATCH_CHECK(get_mods_variant(it) == ModsVariant::NONE);
        }
    }

    CATCH_SECTION("ModsVariant to_string") {
        CATCH_CHECK(to_string(ModsVariant::M_4mC_5mC) == "4mC_5mC");
        CATCH_CHECK(to_string(ModsVariant::M_5mC_5hmC) == "5mC_5hmC");
        CATCH_CHECK(to_string(ModsVariant::M_5mCG) == "5mCG");
        CATCH_CHECK(to_string(ModsVariant::M_5mCG_5hmCG) == "5mCG_5hmCG");
        CATCH_CHECK(to_string(ModsVariant::M_5mC) == "5mC");
        CATCH_CHECK(to_string(ModsVariant::M_m5C) == "m5C");
        CATCH_CHECK(to_string(ModsVariant::M_6mA) == "6mA");
        CATCH_CHECK(to_string(ModsVariant::M_m6A) == "m6A");
        CATCH_CHECK(to_string(ModsVariant::M_m6A_DRACH) == "m6A_DRACH");
        CATCH_CHECK(to_string(ModsVariant::M_inosine_m6A) == "inosine_m6A");
        CATCH_CHECK(to_string(ModsVariant::M_pseU) == "pseU");
        CATCH_CHECK_THROWS_AS(to_string(ModsVariant::NONE), std::logic_error);
    }

    CATCH_SECTION("ModsVariant no duplicates") {
        std::set<std::string> set;
        for (const auto& mod : mods) {
            set.insert(mod.first);
        }
        CATCH_CHECK(set.size() == mods.size());
    }

    CATCH_SECTION("ModsVariant self consistent") {
        for (const auto& mod : mods) {
            CATCH_CHECK(mod.second == get_mods_variant(to_string(mod.second)));
        }
    }
}

CATCH_TEST_CASE(TEST_TAG "  mods_canonical_base_map", TEST_TAG) {
    const auto& mods = mods_canonical_base_map();

    CATCH_SECTION("All expected ModsVariant exist") {
        CATCH_CHECK(mods.at(ModsVariant::M_4mC_5mC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mC_5hmC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mCG) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mCG_5hmCG) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_m5C) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_6mA) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_inosine_m6A) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_m6A) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_m6A_DRACH) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_pseU) == "T");
        CATCH_CHECK(mods.size() == static_cast<size_t>(ModsVariant::NONE));
    }
}

CATCH_TEST_CASE(TEST_TAG "  mods_context_map", TEST_TAG) {
    const auto& mods = mods_context_map();

    CATCH_SECTION("All expected ModsVariant exist") {
        CATCH_CHECK(mods.at(ModsVariant::M_4mC_5mC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mC_5hmC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_5mCG) == "CG");
        CATCH_CHECK(mods.at(ModsVariant::M_5mCG_5hmCG) == "CG");
        CATCH_CHECK(mods.at(ModsVariant::M_5mC) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_m5C) == "C");
        CATCH_CHECK(mods.at(ModsVariant::M_6mA) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_inosine_m6A) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_m6A) == "A");
        CATCH_CHECK(mods.at(ModsVariant::M_m6A_DRACH) == "DRACH");
        CATCH_CHECK(mods.at(ModsVariant::M_pseU) == "T");
        CATCH_CHECK(mods.size() == static_cast<size_t>(ModsVariant::NONE));
    }
}

CATCH_TEST_CASE(TEST_TAG "  ModelVersion enumeration", TEST_TAG) {
    const auto& vers = version_map();

    CATCH_SECTION("ModelVersion to_string") {
        CATCH_CHECK(to_string(ModelVersion::v0_0_0) == "v0.0.0");
        CATCH_CHECK(to_string(ModelVersion::v0_1_0) == "v0.1.0");
        CATCH_CHECK(to_string(ModelVersion::v1_0_0) == "v1.0.0");
        CATCH_CHECK(to_string(ModelVersion::v1_1_0) == "v1.1.0");
        CATCH_CHECK(to_string(ModelVersion::v1_2_0) == "v1.2.0");
        CATCH_CHECK(to_string(ModelVersion::v2_0_0) == "v2.0.0");
        CATCH_CHECK(to_string(ModelVersion::v3_0_0) == "v3.0.0");
        CATCH_CHECK(to_string(ModelVersion::v3_0_1) == "v3.0.1");
        CATCH_CHECK(to_string(ModelVersion::v3_1_0) == "v3.1.0");
        CATCH_CHECK(to_string(ModelVersion::v3_3_0) == "v3.3.0");
        CATCH_CHECK(to_string(ModelVersion::v3_4_0) == "v3.4.0");
        CATCH_CHECK(to_string(ModelVersion::v3_5_0) == "v3.5.0");
        CATCH_CHECK(to_string(ModelVersion::v3_5_2) == "v3.5.2");
        CATCH_CHECK(to_string(ModelVersion::v3_6_0) == "v3.6.0");
        CATCH_CHECK(to_string(ModelVersion::v4_0_0) == "v4.0.0");
        CATCH_CHECK(to_string(ModelVersion::v4_1_0) == "v4.1.0");
        CATCH_CHECK(to_string(ModelVersion::v4_2_0) == "v4.2.0");
        CATCH_CHECK(to_string(ModelVersion::v4_3_0) == "v4.3.0");
        CATCH_CHECK(to_string(ModelVersion::v5_0_0) == "v5.0.0");
        CATCH_CHECK(to_string(ModelVersion::v5_1_0) == "v5.1.0");
        CATCH_CHECK(to_string(ModelVersion::NONE) == "latest");
        CATCH_CHECK(vers.size() == static_cast<size_t>(ModelVersion::NONE) +
                                           1);  // +1 as "NONE" is included in the map
    }

    CATCH_SECTION("ModelVersion no duplicates") {
        std::set<std::string> set;
        for (const auto& ver : vers) {
            set.insert(ver.first);
        }
        CATCH_CHECK(set.size() == vers.size());
    }

    CATCH_SECTION("ModelVersion self consistent") {
        for (const auto& ver : vers) {
            CATCH_CHECK(ver.second == vers.at(to_string(ver.second)));
        }
    }
}

CATCH_TEST_CASE(TEST_TAG "  ModelVariantPair / ModsVariantPair", TEST_TAG) {
    auto model_cp = ModelVariantPair();
    CATCH_CHECK(model_cp.variant == ModelVariant::NONE);
    CATCH_CHECK_FALSE(model_cp.has_variant());
    CATCH_CHECK(model_cp.ver == ModelVersion::NONE);
    CATCH_CHECK_FALSE(model_cp.has_ver());

    model_cp = ModelVariantPair{ModelVariant::AUTO, ModelVersion::v2_0_0};
    CATCH_CHECK(model_cp.variant == ModelVariant::AUTO);
    CATCH_CHECK(model_cp.has_variant());
    CATCH_CHECK(model_cp.ver == ModelVersion::v2_0_0);
    CATCH_CHECK(model_cp.has_ver());

    auto mods_cp = ModsVariantPair();
    CATCH_CHECK(mods_cp.variant == ModsVariant::NONE);
    CATCH_CHECK_FALSE(mods_cp.has_variant());
    CATCH_CHECK(mods_cp.ver == ModelVersion::NONE);
    CATCH_CHECK_FALSE(mods_cp.has_ver());

    mods_cp = ModsVariantPair{ModsVariant::M_5mCG_5hmCG, ModelVersion::v1_1_0};
    CATCH_CHECK(mods_cp.variant == ModsVariant::M_5mCG_5hmCG);
    CATCH_CHECK(mods_cp.has_variant());
    CATCH_CHECK(mods_cp.ver == ModelVersion::v1_1_0);
    CATCH_CHECK(mods_cp.has_ver());
}
