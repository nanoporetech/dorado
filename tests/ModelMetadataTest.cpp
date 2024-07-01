#include "models/metadata.h"

#include <catch2/catch.hpp>

#include <set>
#include <stdexcept>

#define TEST_TAG "[ModelMetadata]"

using namespace dorado::models;

TEST_CASE(TEST_TAG "  ModelVariant enumeration", TEST_TAG) {
    const auto& codes = model_variants_map();

    SECTION("Only expected ModelVariant exist") {
        CHECK(codes.at("auto") == ModelVariant::AUTO);
        CHECK(codes.at("fast") == ModelVariant::FAST);
        CHECK(codes.at("hac") == ModelVariant::HAC);
        CHECK(codes.at("sup") == ModelVariant::SUP);
        CHECK(codes.size() == static_cast<size_t>(ModelVariant::NONE));
    }

    SECTION("ModelVariant get_model_code") {
        CHECK(get_model_variant("auto") == ModelVariant::AUTO);
        CHECK(get_model_variant("fast") == ModelVariant::FAST);
        CHECK(get_model_variant("hac") == ModelVariant::HAC);
        CHECK(get_model_variant("sup") == ModelVariant::SUP);

        for (const auto& it : {"", "foo", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2"}) {
            CHECK(get_model_variant(it) == ModelVariant::NONE);
        }
    }

    SECTION("ModelVariant to_string") {
        CHECK(to_string(ModelVariant::AUTO) == "auto");
        CHECK(to_string(ModelVariant::FAST) == "fast");
        CHECK(to_string(ModelVariant::HAC) == "hac");
        CHECK(to_string(ModelVariant::SUP) == "sup");
        CHECK_THROWS_AS(to_string(ModelVariant::NONE), std::logic_error);
    }

    SECTION("ModelVariant self consistent") {
        for (const auto& code : codes) {
            if (code.second != ModelVariant::NONE) {
                CHECK(code.second == get_model_variant(to_string(code.second)));
            }
        }
    }
}

TEST_CASE(TEST_TAG "  ModsVariant enumeration", TEST_TAG) {
    const auto& mods = mods_variants_map();

    SECTION("Only expected ModsVariant exist") {
        CHECK(mods.at("4mC_5mC") == ModsVariant::M_4mC_5mC);
        CHECK(mods.at("5mC_5hmC") == ModsVariant::M_5mC_5hmC);
        CHECK(mods.at("5mCG") == ModsVariant::M_5mCG);
        CHECK(mods.at("5mCG_5hmCG") == ModsVariant::M_5mCG_5hmCG);
        CHECK(mods.at("5mC") == ModsVariant::M_5mC);
        CHECK(mods.at("6mA") == ModsVariant::M_6mA);
        CHECK(mods.at("m6A") == ModsVariant::M_m6A);
        CHECK(mods.at("m6A_DRACH") == ModsVariant::M_m6A_DRACH);
        CHECK(mods.at("pseU") == ModsVariant::M_pseU);
        CHECK(mods.size() == static_cast<size_t>(ModsVariant::NONE));
    }

    SECTION("ModsVariant get_mods_code") {
        CHECK(get_mods_variant("4mC_5mC") == ModsVariant::M_4mC_5mC);
        CHECK(get_mods_variant("5mC_5hmC") == ModsVariant::M_5mC_5hmC);
        CHECK(get_mods_variant("5mCG") == ModsVariant::M_5mCG);
        CHECK(get_mods_variant("5mCG_5hmCG") == ModsVariant::M_5mCG_5hmCG);
        CHECK(get_mods_variant("5mC") == ModsVariant::M_5mC);
        CHECK(get_mods_variant("6mA") == ModsVariant::M_6mA);
        CHECK(get_mods_variant("m6A") == ModsVariant::M_m6A);
        CHECK(get_mods_variant("m6A_DRACH") == ModsVariant::M_m6A_DRACH);
        CHECK(get_mods_variant("pseU") == ModsVariant::M_pseU);
        for (const auto& it : {"", "foo", "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2"}) {
            CHECK(get_mods_variant(it) == ModsVariant::NONE);
        }
    }

    SECTION("ModsVariant to_string") {
        CHECK(to_string(ModsVariant::M_4mC_5mC) == "4mC_5mC");
        CHECK(to_string(ModsVariant::M_5mC_5hmC) == "5mC_5hmC");
        CHECK(to_string(ModsVariant::M_5mCG) == "5mCG");
        CHECK(to_string(ModsVariant::M_5mCG_5hmCG) == "5mCG_5hmCG");
        CHECK(to_string(ModsVariant::M_5mC) == "5mC");
        CHECK(to_string(ModsVariant::M_6mA) == "6mA");
        CHECK(to_string(ModsVariant::M_m6A_DRACH) == "m6A_DRACH");
        CHECK(to_string(ModsVariant::M_m6A) == "m6A");
        CHECK(to_string(ModsVariant::M_pseU) == "pseU");
        CHECK_THROWS_AS(to_string(ModsVariant::NONE), std::logic_error);
    }

    SECTION("ModsVariant no duplicates") {
        std::set<std::string> set;
        for (const auto& mod : mods) {
            set.insert(mod.first);
        }
        CHECK(set.size() == mods.size());
    }

    SECTION("ModsVariant self consistent") {
        for (const auto& mod : mods) {
            CHECK(mod.second == get_mods_variant(to_string(mod.second)));
        }
    }
}

TEST_CASE(TEST_TAG "  mods_canonical_base_map", TEST_TAG) {
    const auto& mods = mods_canonical_base_map();

    SECTION("All expected ModsVariant exist") {
        CHECK(mods.at(ModsVariant::M_4mC_5mC) == "C");
        CHECK(mods.at(ModsVariant::M_5mC_5hmC) == "C");
        CHECK(mods.at(ModsVariant::M_5mCG) == "C");
        CHECK(mods.at(ModsVariant::M_5mCG_5hmCG) == "C");
        CHECK(mods.at(ModsVariant::M_5mC) == "C");
        CHECK(mods.at(ModsVariant::M_6mA) == "A");
        CHECK(mods.at(ModsVariant::M_m6A) == "A");
        CHECK(mods.at(ModsVariant::M_m6A_DRACH) == "A");
        CHECK(mods.at(ModsVariant::M_pseU) == "T");
        CHECK(mods.size() == static_cast<size_t>(ModsVariant::NONE));
    }
}

TEST_CASE(TEST_TAG "  ModelVersion enumeration", TEST_TAG) {
    const auto& vers = version_map();

    SECTION("ModelVersion to_string") {
        CHECK(to_string(ModelVersion::v0_0_0) == "v0.0.0");
        CHECK(to_string(ModelVersion::v0_1_0) == "v0.1.0");
        CHECK(to_string(ModelVersion::v1_0_0) == "v1.0.0");
        CHECK(to_string(ModelVersion::v1_1_0) == "v1.1.0");
        CHECK(to_string(ModelVersion::v1_2_0) == "v1.2.0");
        CHECK(to_string(ModelVersion::v2_0_0) == "v2.0.0");
        CHECK(to_string(ModelVersion::v3_0_0) == "v3.0.0");
        CHECK(to_string(ModelVersion::v3_0_1) == "v3.0.1");
        CHECK(to_string(ModelVersion::v3_1_0) == "v3.1.0");
        CHECK(to_string(ModelVersion::v3_3_0) == "v3.3.0");
        CHECK(to_string(ModelVersion::v3_4_0) == "v3.4.0");
        CHECK(to_string(ModelVersion::v3_5_0) == "v3.5.0");
        CHECK(to_string(ModelVersion::v3_5_2) == "v3.5.2");
        CHECK(to_string(ModelVersion::v3_6_0) == "v3.6.0");
        CHECK(to_string(ModelVersion::v4_0_0) == "v4.0.0");
        CHECK(to_string(ModelVersion::v4_1_0) == "v4.1.0");
        CHECK(to_string(ModelVersion::v4_2_0) == "v4.2.0");
        CHECK(to_string(ModelVersion::v4_3_0) == "v4.3.0");
        CHECK(to_string(ModelVersion::NONE) == "latest");
        CHECK(vers.size() ==
              static_cast<size_t>(ModelVersion::NONE) + 1);  // +1 as "NONE" is included in the map
    }

    SECTION("ModelVersion no duplicates") {
        std::set<std::string> set;
        for (const auto& ver : vers) {
            set.insert(ver.first);
        }
        CHECK(set.size() == vers.size());
    }

    SECTION("ModelVersion self consistent") {
        for (const auto& ver : vers) {
            CHECK(ver.second == vers.at(to_string(ver.second)));
        }
    }
}

TEST_CASE(TEST_TAG "  ModelVariantPair / ModsVariantPair", TEST_TAG) {
    auto model_cp = ModelVariantPair();
    CHECK(model_cp.variant == ModelVariant::NONE);
    CHECK_FALSE(model_cp.has_variant());
    CHECK(model_cp.ver == ModelVersion::NONE);
    CHECK_FALSE(model_cp.has_ver());

    model_cp = ModelVariantPair{ModelVariant::AUTO, ModelVersion::v2_0_0};
    CHECK(model_cp.variant == ModelVariant::AUTO);
    CHECK(model_cp.has_variant());
    CHECK(model_cp.ver == ModelVersion::v2_0_0);
    CHECK(model_cp.has_ver());

    auto mods_cp = ModsVariantPair();
    CHECK(mods_cp.variant == ModsVariant::NONE);
    CHECK_FALSE(mods_cp.has_variant());
    CHECK(mods_cp.ver == ModelVersion::NONE);
    CHECK_FALSE(mods_cp.has_ver());

    mods_cp = ModsVariantPair{ModsVariant::M_5mCG_5hmCG, ModelVersion::v1_1_0};
    CHECK(mods_cp.variant == ModsVariant::M_5mCG_5hmCG);
    CHECK(mods_cp.has_variant());
    CHECK(mods_cp.ver == ModelVersion::v1_1_0);
    CHECK(mods_cp.has_ver());
}
