#include "models/kits.h"
#include "models/metadata.h"
#include "utils/string_utils.h"

#include <catch2/catch.hpp>

#include <set>
#include <stdexcept>

#define TEST_TAG "[ModelKits]"

using namespace dorado::models;

TEST_CASE(TEST_TAG " FlowcellCode enumeration", TEST_TAG) {
    const auto& fcs = flowcell_codes();

    SECTION("FlowcellCode to_string") {
        CHECK(to_string(Flowcell::FLO_FLG001) == "FLO-FLG001");
        CHECK(to_string(Flowcell::FLO_FLG114) == "FLO-FLG114");
        CHECK(to_string(Flowcell::FLO_FLG114HD) == "FLO-FLG114HD");
        CHECK(to_string(Flowcell::FLO_MIN004RA) == "FLO-MIN004RA");
        CHECK(to_string(Flowcell::FLO_MIN106) == "FLO-MIN106");
        CHECK(to_string(Flowcell::FLO_MIN107) == "FLO-MIN107");
        CHECK(to_string(Flowcell::FLO_MIN112) == "FLO-MIN112");
        CHECK(to_string(Flowcell::FLO_MIN114) == "FLO-MIN114");
        CHECK(to_string(Flowcell::FLO_MIN114HD) == "FLO-MIN114HD");
        CHECK(to_string(Flowcell::FLO_MINSP6) == "FLO-MINSP6");
        CHECK(to_string(Flowcell::FLO_PRO001) == "FLO-PRO001");
        CHECK(to_string(Flowcell::FLO_PRO002) == "FLO-PRO002");
        CHECK(to_string(Flowcell::FLO_PRO002_ECO) == "FLO-PRO002-ECO");
        CHECK(to_string(Flowcell::FLO_PRO002M) == "FLO-PRO002M");
        CHECK(to_string(Flowcell::FLO_PRO004RA) == "FLO-PRO004RA");
        CHECK(to_string(Flowcell::FLO_PRO112) == "FLO-PRO112");
        CHECK(to_string(Flowcell::FLO_PRO114) == "FLO-PRO114");
        CHECK(to_string(Flowcell::FLO_PRO114HD) == "FLO-PRO114HD");
        CHECK(to_string(Flowcell::FLO_PRO114M) == "FLO-PRO114M");
        CHECK(to_string(Flowcell::UNKNOWN) == "__UNKNOWN_FLOWCELL__");
        CHECK(fcs.size() == static_cast<size_t>(Flowcell::UNKNOWN) + 1);
    }

    SECTION("FlowcellCode self consistent") {
        for (const auto& fc : fcs) {
            CHECK(fc.first == flowcell_code(to_string(fc.first)));
        }
    }

    SECTION("FlowcellCodes contain no underscores") {
        for (const auto& fc : fcs) {
            if (fc.first == Flowcell::UNKNOWN) {
                continue;
            }
            const auto& name = fc.second.name;
            CHECK(std::count(name.begin(), name.end(), '_') == 0);
        }
    }

    SECTION("FlowcellCodes no duplicates") {
        std::set<std::string> set;
        for (const auto& fc : fcs) {
            set.insert(fc.second.name);
        }
        CHECK(set.size() == fcs.size());
    }
}

TEST_CASE(TEST_TAG "  KitCode enumeration", TEST_TAG) {
    const auto& kits = kit_codes();

    SECTION("KitCode to_string") {
        CHECK(to_string(KitCode::SQK_APK114) == "SQK-APK114");
        CHECK(to_string(KitCode::SQK_CS9109) == "SQK-CS9109");
        CHECK(to_string(KitCode::SQK_DCS108) == "SQK-DCS108");
        CHECK(to_string(KitCode::SQK_DCS109) == "SQK-DCS109");
        CHECK(to_string(KitCode::SQK_LRK001) == "SQK-LRK001");
        CHECK(to_string(KitCode::SQK_LSK108) == "SQK-LSK108");
        CHECK(to_string(KitCode::SQK_LSK109) == "SQK-LSK109");
        CHECK(to_string(KitCode::SQK_LSK109_XL) == "SQK-LSK109-XL");
        CHECK(to_string(KitCode::SQK_LSK110) == "SQK-LSK110");
        CHECK(to_string(KitCode::SQK_LSK110_XL) == "SQK-LSK110-XL");
        CHECK(to_string(KitCode::SQK_LSK111) == "SQK-LSK111");
        CHECK(to_string(KitCode::SQK_LSK111_XL) == "SQK-LSK111-XL");
        CHECK(to_string(KitCode::SQK_LSK112) == "SQK-LSK112");
        CHECK(to_string(KitCode::SQK_LSK112_XL) == "SQK-LSK112-XL");
        CHECK(to_string(KitCode::SQK_LSK114) == "SQK-LSK114");
        CHECK(to_string(KitCode::SQK_LSK114_260) == "SQK-LSK114-260");
        CHECK(to_string(KitCode::SQK_LSK114_XL) == "SQK-LSK114-XL");
        CHECK(to_string(KitCode::SQK_LSK114_XL_260) == "SQK--260LSK114-XL-260");
        CHECK(to_string(KitCode::SQK_LWP001) == "SQK-LWP001");
        CHECK(to_string(KitCode::SQK_PCS108) == "SQK-PCS108");
        CHECK(to_string(KitCode::SQK_PCS109) == "SQK-PCS109");
        CHECK(to_string(KitCode::SQK_PCS111) == "SQK-PCS111");
        CHECK(to_string(KitCode::SQK_PCS114) == "SQK-PCS114");
        CHECK(to_string(KitCode::SQK_PCS114_260) == "SQK-PCS114-260");
        CHECK(to_string(KitCode::SQK_PSK004) == "SQK-PSK004");
        CHECK(to_string(KitCode::SQK_RAD002) == "SQK-RAD002");
        CHECK(to_string(KitCode::SQK_RAD003) == "SQK-RAD003");
        CHECK(to_string(KitCode::SQK_RAD004) == "SQK-RAD004");
        CHECK(to_string(KitCode::SQK_RAD112) == "SQK-RAD112");
        CHECK(to_string(KitCode::SQK_RAD114) == "SQK-RAD114");
        CHECK(to_string(KitCode::SQK_RAD114_260) == "SQK-RAD114-260");
        CHECK(to_string(KitCode::SQK_RAS201) == "SQK-RAS201");
        CHECK(to_string(KitCode::SQK_RLI001) == "SQK-RLI001");
        CHECK(to_string(KitCode::SQK_RNA001) == "SQK-RNA001");
        CHECK(to_string(KitCode::SQK_RNA002) == "SQK-RNA002");
        CHECK(to_string(KitCode::SQK_RNA004) == "SQK-RNA004");
        CHECK(to_string(KitCode::SQK_ULK001) == "SQK-ULK001");
        CHECK(to_string(KitCode::SQK_ULK114) == "SQK-ULK114");
        CHECK(to_string(KitCode::SQK_ULK114_260) == "SQK-ULK114-260");
        CHECK(to_string(KitCode::VSK_VBK001) == "VSK-VBK001");
        CHECK(to_string(KitCode::VSK_VSK001) == "VSK-VSK001");
        CHECK(to_string(KitCode::VSK_VSK003) == "VSK-VSK003");
        CHECK(to_string(KitCode::VSK_VSK004) == "VSK-VSK004");
        // Barcoding
        CHECK(to_string(KitCode::SQK_16S024) == "SQK-16S024");
        CHECK(to_string(KitCode::SQK_16S114_24) == "SQK-16S114-24");
        CHECK(to_string(KitCode::SQK_16S114_24_260) == "SQK-16S114-24-260");
        CHECK(to_string(KitCode::SQK_LWB001) == "SQK-LWB001");
        CHECK(to_string(KitCode::SQK_MLK111_96_XL) == "SQK-MLK111-96-XL");
        CHECK(to_string(KitCode::SQK_MLK114_96_XL) == "SQK-MLK114-96-XL");
        CHECK(to_string(KitCode::SQK_MLK114_96_XL_260) == "SQK-MLK114-96-XL-260");
        CHECK(to_string(KitCode::SQK_NBD111_24) == "SQK-NBD111-24");
        CHECK(to_string(KitCode::SQK_NBD111_96) == "SQK-NBD111-96");
        CHECK(to_string(KitCode::SQK_NBD112_24) == "SQK-NBD112-24");
        CHECK(to_string(KitCode::SQK_NBD112_96) == "SQK-NBD112-96");
        CHECK(to_string(KitCode::SQK_NBD114_24) == "SQK-NBD114-24");
        CHECK(to_string(KitCode::SQK_NBD114_24_260) == "SQK-NBD114-24-260");
        CHECK(to_string(KitCode::SQK_NBD114_96) == "SQK-NBD114-96");
        CHECK(to_string(KitCode::SQK_NBD114_96_260) == "SQK-NBD114-96-260");
        CHECK(to_string(KitCode::SQK_PBK004) == "SQK-PBK004");
        CHECK(to_string(KitCode::SQK_PCB109) == "SQK-PCB109");
        CHECK(to_string(KitCode::SQK_PCB110) == "SQK-PCB110");
        CHECK(to_string(KitCode::SQK_PCB111_24) == "SQK-PCB111-24");
        CHECK(to_string(KitCode::SQK_PCB114_24) == "SQK-PCB114-24");
        CHECK(to_string(KitCode::SQK_PCB114_24_260) == "SQK-PCB114-24-260");
        CHECK(to_string(KitCode::SQK_RAB201) == "SQK-RAB201");
        CHECK(to_string(KitCode::SQK_RAB204) == "SQK-RAB204");
        CHECK(to_string(KitCode::SQK_RBK001) == "SQK-RBK001");
        CHECK(to_string(KitCode::SQK_RBK004) == "SQK-RBK004");
        CHECK(to_string(KitCode::SQK_RBK110_96) == "SQK-RBK110-96");
        CHECK(to_string(KitCode::SQK_RBK111_24) == "SQK-RBK111-24");
        CHECK(to_string(KitCode::SQK_RBK111_96) == "SQK-RBK111-96");
        CHECK(to_string(KitCode::SQK_RBK114_24) == "SQK-RBK114-24");
        CHECK(to_string(KitCode::SQK_RBK114_24_260) == "SQK-RBK114-24-260");
        CHECK(to_string(KitCode::SQK_RBK114_96) == "SQK-RBK114-96");
        CHECK(to_string(KitCode::SQK_RBK114_96_260) == "SQK-RBK114-96-260");
        CHECK(to_string(KitCode::SQK_RLB001) == "SQK-RLB001");
        CHECK(to_string(KitCode::SQK_RPB004) == "SQK-RPB004");
        CHECK(to_string(KitCode::SQK_RPB114_24) == "SQK-RPB114-24");
        CHECK(to_string(KitCode::SQK_RPB114_24_260) == "SQK-RPB114-24-260");
        CHECK(to_string(KitCode::VSK_PTC001) == "VSK-PTC001");
        CHECK(to_string(KitCode::VSK_VMK001) == "VSK-VMK001");
        CHECK(to_string(KitCode::VSK_VMK004) == "VSK-VMK004");
        CHECK(to_string(KitCode::VSK_VPS001) == "VSK-VPS001");

        CHECK(to_string(KitCode::UNKNOWN) == "__UNKNOWN_KIT__");
        CHECK(kits.size() == static_cast<size_t>(KitCode::UNKNOWN) + 1);
    }

    SECTION("KitCode contain no underscores") {
        for (const auto& kit : kits) {
            if (kit.first == KitCode::UNKNOWN) {
                continue;
            }
            const auto& name = kit.second.name;
            CHECK(std::count(name.begin(), name.end(), '_') == 0);
        }
    }

    SECTION("KitCode motor speed") {
        using namespace dorado::utils;
        for (const auto& kit : kits) {
            const auto& name = kit.second.name;
            const auto& speed = kit.second.speed;
            if (name.find("112") != std::string::npos) {
                CHECK(speed == 250);
            } else if (ends_with(name, "-260") || ends_with(name, "APK114")) {
                CHECK(speed == 260);
            } else if (ends_with(name, "RNA002")) {
                CHECK(speed == 70);
            } else if (ends_with(name, "RNA004")) {
                CHECK(speed == 130);
            } else if (ends_with(name, "__UNKNOWN_KIT__")) {
                continue;
            } else {
                CHECK(kit.second.speed == 400);
            }
        }
    }
}
