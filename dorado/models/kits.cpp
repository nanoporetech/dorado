#include "kits.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace dorado::models {

using FC = Flowcell;
using KC = KitCode;

template <typename Code, typename Info>
Code get_code(const std::string& str,
              Code default_value,
              const std::unordered_map<Code, Info>& codes) {
    auto s = str;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    auto it = std::find_if(std::begin(codes), std::end(codes),
                           [&s](auto&& kv) { return kv.second.name == s; });

    if (it == std::end(codes)) {
        return default_value;
    }
    return it->first;
}

template <typename Code, typename Info>
Info get_info(const Code& code,
              const std::string& description,
              const std::unordered_map<Code, Info>& codes) {
    auto it = codes.find(code);
    if (it == std::end(codes)) {
        throw std::logic_error("Unknown " + description +
                               " enum: " + std::to_string(static_cast<int>(code)));
    }
    return it->second;
}

namespace flowcell {

// clang-format off
// NEED TO ADD HIGH DUPLEX FCS
const std::unordered_map<Flowcell, FlowcellInfo> codes_map = {
        {FC::FLO_FLG001,    {"FLO-FLG001",    }}, 
        {FC::FLO_FLG114,    {"FLO-FLG114",    }},
        {FC::FLO_FLG114HD,  {"FLO-FLG114HD",  }},
        {FC::FLO_MIN004RA,  {"FLO-MIN004RA",  }}, 
        {FC::FLO_MIN106,    {"FLO-MIN106",    }},
        {FC::FLO_MIN107,    {"FLO-MIN107",    }}, 
        {FC::FLO_MIN112,    {"FLO-MIN112",    }},
        {FC::FLO_MIN114,    {"FLO-MIN114",    }},
        {FC::FLO_MIN114HD,  {"FLO-MIN114HD",  }},
        {FC::FLO_MINSP6,    {"FLO-MINSP6",    }}, 
        {FC::FLO_PRO001,    {"FLO-PRO001",    }},
        {FC::FLO_PRO002,    {"FLO-PRO002",    }}, 
        {FC::FLO_PRO002_ECO,{"FLO-PRO002-ECO",}},
        {FC::FLO_PRO002M,   {"FLO-PRO002M",   }}, 
        {FC::FLO_PRO004RA,  {"FLO-PRO004RA",  }},
        {FC::FLO_PRO112,    {"FLO-PRO112",    }}, 
        {FC::FLO_PRO114,    {"FLO-PRO114",    }}, 
        {FC::FLO_PRO114M,   {"FLO-PRO114M",   }},
        {FC::FLO_PRO114HD,  {"FLO-PRO114HD",  }}, 
        {FC::FLO_PRO114M,   {"FLO-PRO114M",   }},
        {FC::UNKNOWN,       {"__UNKNOWN_FLOWCELL__", }}, 
};
// clang-format on

}  // namespace flowcell

const std::unordered_map<Flowcell, FlowcellInfo>& flowcell_codes() { return flowcell::codes_map; }

Flowcell flowcell_code(const std::string& code) {
    return get_code(code, Flowcell::UNKNOWN, flowcell_codes());
}

FlowcellInfo flowcell_info(const Flowcell& fc) {
    return get_info(fc, "flowcell_product_code", flowcell_codes());
}

std::string to_string(const Flowcell& fc) { return flowcell_info(fc).name; }

namespace kit {

const ProtocolInfo not_rapid{true, RapidChemistry::NONE};
const ProtocolInfo rapid_v1{true, RapidChemistry::V1};

const std::unordered_map<KitCode, KitInfo> codes_map = {
        {KC::SQK_APK114, {"SQK-APK114", 260}},
        {KC::SQK_CS9109, {"SQK-CS9109", 400}},
        {KC::SQK_DCS108, {"SQK-DCS108", 400}},
        {KC::SQK_DCS109, {"SQK-DCS109", 400}},
        {KC::SQK_LRK001, {"SQK-LRK001", 400}},
        {KC::SQK_LSK108, {"SQK-LSK108", 400}},
        {KC::SQK_LSK109, {"SQK-LSK109", 400}},
        {KC::SQK_LSK109_XL, {"SQK-LSK109-XL", 400}},
        {KC::SQK_LSK110, {"SQK-LSK110", 400}},
        {KC::SQK_LSK110_XL, {"SQK-LSK110-XL", 400}},
        {KC::SQK_LSK111, {"SQK-LSK111", 400}},
        {KC::SQK_LSK111_XL, {"SQK-LSK111-XL", 400}},
        {KC::SQK_LSK112, {"SQK-LSK112", 250}},
        {KC::SQK_LSK112_XL, {"SQK-LSK112-XL", 250}},
        {KC::SQK_LSK114, {"SQK-LSK114", 400}},
        {KC::SQK_LSK114_260, {"SQK-LSK114-260", 260}},
        {KC::SQK_LSK114_XL, {"SQK-LSK114-XL", 400}},
        {KC::SQK_LSK114_XL_260, {"SQK--260LSK114-XL-260", 260}},
        {KC::SQK_LWP001, {"SQK-LWP001", 400}},
        {KC::SQK_PCS108, {"SQK-PCS108", 400}},
        {KC::SQK_PCS109, {"SQK-PCS109", 400}},
        {KC::SQK_PCS111, {"SQK-PCS111", 400}},
        {KC::SQK_PCS114, {"SQK-PCS114", 400}},
        {KC::SQK_PCS114_260, {"SQK-PCS114-260", 260}},
        {KC::SQK_PSK004, {"SQK-PSK004", 400}},
        {KC::SQK_RAD002, {"SQK-RAD002", 400}},
        {KC::SQK_RAD003, {"SQK-RAD003", 400}},
        {KC::SQK_RAD004, {"SQK-RAD004", 400}},
        {KC::SQK_RAD112, {"SQK-RAD112", 250}},
        {KC::SQK_RAD114, {"SQK-RAD114", 400}},
        {KC::SQK_RAD114_260, {"SQK-RAD114-260", 260}},
        {KC::SQK_RAS201, {"SQK-RAS201", 400}},
        {KC::SQK_RLI001, {"SQK-RLI001", 400}},
        {KC::SQK_RNA001, {"SQK-RNA001", 400}},
        {KC::SQK_RNA002, {"SQK-RNA002", 70}},
        {KC::SQK_RNA004, {"SQK-RNA004", 130}},
        {KC::SQK_RNA004_XL, {"SQK-RNA004-XL", 130}},
        {KC::SQK_ULK001, {"SQK-ULK001", 400}},
        {KC::SQK_ULK114, {"SQK-ULK114", 400}},
        {KC::SQK_ULK114_260, {"SQK-ULK114-260", 260}},
        {KC::VSK_VBK001, {"VSK-VBK001", 400}},
        {KC::VSK_VSK001, {"VSK-VSK001", 400}},
        {KC::VSK_VSK003, {"VSK-VSK003", 400}},
        {KC::VSK_VSK004, {"VSK-VSK004", 400}},

        // Barcoding Kits
        {KC::SQK_16S024, {"SQK-16S024", 400, not_rapid}},
        {KC::SQK_16S114_24, {"SQK-16S114-24", 400, not_rapid}},
        {KC::SQK_16S114_24_260, {"SQK-16S114-24-260", 260, not_rapid}},
        {KC::SQK_MAB114_24, {"SQK-MAB114-24", 400, not_rapid}},
        {KC::SQK_LWB001, {"SQK-LWB001", 400, not_rapid}},
        {KC::SQK_MLK111_96_XL, {"SQK-MLK111-96-XL", 400, not_rapid}},
        {KC::SQK_MLK114_96_XL, {"SQK-MLK114-96-XL", 400, not_rapid}},
        {KC::SQK_MLK114_96_XL_260, {"SQK-MLK114-96-XL-260", 260, not_rapid}},
        {KC::SQK_NBD111_24, {"SQK-NBD111-24", 400, not_rapid}},
        {KC::SQK_NBD111_96, {"SQK-NBD111-96", 400, not_rapid}},
        {KC::SQK_NBD112_24, {"SQK-NBD112-24", 250, not_rapid}},
        {KC::SQK_NBD112_96, {"SQK-NBD112-96", 250, not_rapid}},
        {KC::SQK_NBD114_24, {"SQK-NBD114-24", 400, not_rapid}},
        {KC::SQK_NBD114_24_260, {"SQK-NBD114-24-260", 260, not_rapid}},
        {KC::SQK_NBD114_96, {"SQK-NBD114-96", 400, not_rapid}},
        {KC::SQK_NBD114_96_260, {"SQK-NBD114-96-260", 260, not_rapid}},
        {KC::SQK_PBK004, {"SQK-PBK004", 400, not_rapid}},
        {KC::SQK_PCB109, {"SQK-PCB109", 400, not_rapid}},
        {KC::SQK_PCB110, {"SQK-PCB110", 400, not_rapid}},
        {KC::SQK_PCB111_24, {"SQK-PCB111-24", 400, not_rapid}},
        {KC::SQK_PCB114_24, {"SQK-PCB114-24", 400, not_rapid}},
        {KC::SQK_PCB114_24_260, {"SQK-PCB114-24-260", 260, not_rapid}},
        {KC::SQK_RAB201, {"SQK-RAB201", 400, not_rapid}},
        {KC::SQK_RAB204, {"SQK-RAB204", 400, not_rapid}},
        {KC::SQK_RBK001, {"SQK-RBK001", 400, rapid_v1}},
        {KC::SQK_RBK004, {"SQK-RBK004", 400, rapid_v1}},
        {KC::SQK_RBK110_96, {"SQK-RBK110-96", 400, rapid_v1}},
        {KC::SQK_RBK111_24, {"SQK-RBK111-24", 400, rapid_v1}},
        {KC::SQK_RBK111_96, {"SQK-RBK111-96", 400, rapid_v1}},
        {KC::SQK_RBK114_24, {"SQK-RBK114-24", 400, rapid_v1}},
        {KC::SQK_RBK114_24_260, {"SQK-RBK114-24-260", 260, rapid_v1}},
        {KC::SQK_RBK114_96, {"SQK-RBK114-96", 400, rapid_v1}},
        {KC::SQK_RBK114_96_260, {"SQK-RBK114-96-260", 260, rapid_v1}},
        {KC::SQK_RLB001, {"SQK-RLB001", 400, not_rapid}},
        {KC::SQK_RPB004, {"SQK-RPB004", 400, not_rapid}},
        {KC::SQK_RPB114_24, {"SQK-RPB114-24", 400, not_rapid}},
        {KC::SQK_RPB114_24_260, {"SQK-RPB114-24-260", 260, not_rapid}},
        {KC::VSK_PTC001, {"VSK-PTC001", 400, not_rapid}},
        {KC::VSK_VMK001, {"VSK-VMK001", 400, not_rapid}},
        {KC::VSK_VMK004, {"VSK-VMK004", 400, not_rapid}},
        {KC::VSK_VPS001, {"VSK-VPS001", 400, not_rapid}},

        {KC::UNKNOWN, {"__UNKNOWN_KIT__", 1, {false, RapidChemistry::UNKNOWN}}},
};

}  // namespace kit

const std::unordered_map<KitCode, KitInfo>& kit_codes() { return kit::codes_map; }

KitCode kit_code(const std::string& kit) { return get_code(kit, KitCode::UNKNOWN, kit_codes()); }
KitInfo kit_info(const KitCode& kit) { return get_info(kit, "sequencing_kit", kit_codes()); }
std::string to_string(const KitCode& kit) { return kit_info(kit).name; }

namespace kit_sets {

namespace kit14 {
// Kit14 ~ R10.4.1 e8.2
const std::vector<FC> flowcells = {
        FC::FLO_MIN114,
        FC::FLO_FLG114,
        FC::FLO_PRO114,
        FC::FLO_PRO114M,
};

const std::vector<KC> kits_400bps = {
        KC::SQK_LSK114,
        KC::SQK_LSK114_XL,
        KC::SQK_ULK114,
        KC::SQK_RAD114,
        KC::SQK_PCS114,
        // Barcoding
        KC::SQK_NBD114_24,
        KC::SQK_NBD114_96,
        KC::SQK_RBK114_24,
        KC::SQK_RBK114_96,
        KC::SQK_RPB114_24,
        KC::SQK_MLK114_96_XL,
        KC::SQK_16S114_24,
        KC::SQK_MAB114_24,
        KC::SQK_PCB114_24,
};

const std::vector<KC> kits_260bps = {
        KC::SQK_LSK114_260,
        KC::SQK_LSK114_XL_260,
        KC::SQK_ULK114_260,
        KC::SQK_RAD114_260,
        KC::SQK_PCS114_260,
        // Barcoding
        KC::SQK_NBD114_24_260,
        KC::SQK_NBD114_96_260,
        KC::SQK_RBK114_24_260,
        KC::SQK_RBK114_96_260,
        KC::SQK_RPB114_24_260,
        KC::SQK_MLK114_96_XL_260,
        KC::SQK_16S114_24_260,
        KC::SQK_PCB114_24_260,
};

const std::vector<FC> flowcells_hd = {
        FC::FLO_MIN114HD,
        FC::FLO_FLG114HD,
        FC::FLO_PRO114HD,
};

const std::vector<KC> kits_400bps_5khz_hd = {
        KC::SQK_LSK114,
        KC::SQK_LSK114_XL,
};

const KitSets sets_400bps_5khz = {
        {kit14::flowcells, kit14::kits_400bps},
        {kit14::flowcells_hd, kit14::kits_400bps_5khz_hd},
};
const KitSets sets_400bps = {{kit14::flowcells, kit14::kits_400bps}};
const KitSets sets_260bps = {{kit14::flowcells, kit14::kits_260bps}};

const KitSets sets_apk = {{kit14::flowcells, {KC::SQK_APK114}}};
}  // namespace kit14

namespace kit10 {
// Kit10 ~ R9.4.1 e8

const std::vector<FC> flowcells = {
        FC::FLO_FLG001, FC::FLO_MIN106,     FC::FLO_MINSP6,  FC::FLO_PRO001,
        FC::FLO_PRO002, FC::FLO_PRO002_ECO, FC::FLO_PRO002M,
};

const std::vector<KC> kits = {
        KC::SQK_CS9109,
        KC::SQK_DCS108,
        KC::SQK_DCS109,
        KC::SQK_LRK001,
        KC::SQK_LSK108,
        KC::SQK_LSK109,
        KC::SQK_LSK109_XL,
        KC::SQK_LSK110,
        KC::SQK_LSK110_XL,
        KC::SQK_LSK111,
        KC::SQK_LSK111_XL,
        KC::SQK_LWP001,
        KC::SQK_PCS108,
        KC::SQK_PCS109,
        KC::SQK_PCS111,
        KC::SQK_PSK004,
        KC::SQK_RAD002,
        KC::SQK_RAD003,
        KC::SQK_RAD004,
        KC::SQK_RAS201,
        KC::SQK_RLI001,
        KC::SQK_ULK001,
        KC::VSK_VBK001,
        KC::VSK_VSK001,
        KC::VSK_VSK003,
        KC::VSK_VSK004,
        // Barcoding
        KC::SQK_16S024,
        KC::SQK_MLK111_96_XL,
        KC::SQK_NBD111_24,
        KC::SQK_NBD111_96,
        KC::SQK_PCB109,
        KC::SQK_PCB110,
        KC::SQK_PCB111_24,
        KC::SQK_RBK001,
        KC::SQK_RBK004,
        KC::SQK_RBK110_96,
        KC::SQK_RBK111_24,
        KC::SQK_RBK111_96,
        KC::SQK_RLB001,
        KC::SQK_LWB001,
        KC::SQK_PBK004,
        KC::SQK_RAB201,
        KC::SQK_RAB204,
        KC::SQK_RPB004,
        KC::VSK_PTC001,
        KC::VSK_VMK001,
        KC::VSK_VPS001,
        KC::VSK_VMK004,
};

const KitSets sets = {{kit10::flowcells, kit10::kits}};
}  // namespace kit10

namespace rna002 {
// RNA002 Flowcells and Kits
const std::vector<FC> flowcells = {
        FC::FLO_FLG001, FC::FLO_MIN106, FC::FLO_MINSP6,     FC::FLO_MIN107,
        FC::FLO_PRO001, FC::FLO_PRO002, FC::FLO_PRO002_ECO, FC::FLO_PRO002M,
};
const std::vector<KC> kits = {KC::SQK_RNA002};
const KitSets sets = {{flowcells, kits}};
}  // namespace rna002

namespace rna004 {
// RNA004 Flowcells and Kits
const std::vector<FC> flowcells = {FC::FLO_PRO004RA, FC::FLO_MIN004RA};
const std::vector<KC> kits = {KC::SQK_RNA004, KC::SQK_RNA004_XL};
const KitSets sets = {{rna004::flowcells, rna004::kits}};
}  // namespace rna004

}  // namespace kit_sets

namespace chemistry {

const std::unordered_map<SampleType, SampleTypeInfo> sample_type_map = {
        {SampleType::DNA, {"DNA"}},
        {SampleType::RNA002, {"RNA002"}},
        {SampleType::RNA004, {"RNA004"}},
};

/*
Mapping of Chemistry to the complete set of sequencing kits associated with that chemistry.
The cross product of a chemistry's set of flowcells and kits is generated at runtime
*/
const std::unordered_map<Chemistry, ChemistryKits> kit_map = {
        {Chemistry::UNKNOWN, {"__UNKNOWN_CHEMISTRY__", 1, SampleType::DNA, {}}},
        {Chemistry::DNA_R9_4_1_E8, {"dna_r9.4.1_e8", 4000, SampleType::DNA, kit_sets::kit10::sets}},
        {Chemistry::DNA_R10_4_1_E8_2_260BPS,
         {"dna_r10.4.1_e8.2_260bps", 4000, SampleType::DNA, kit_sets::kit14::sets_260bps}},
        {Chemistry::DNA_R10_4_1_E8_2_APK_5KHZ,
         {"dna_r10.4.1_e8.2_apk_5khz", 5000, SampleType::DNA, kit_sets::kit14::sets_apk}},
        {Chemistry::DNA_R10_4_1_E8_2_400BPS_4KHZ,
         {"dna_r10.4.1_e8.2_400bps_4khz", 4000, SampleType::DNA, kit_sets::kit14::sets_400bps}},
        {Chemistry::DNA_R10_4_1_E8_2_400BPS_5KHZ,
         {"dna_r10.4.1_e8.2_400bps_5khz", 5000, SampleType::DNA,
          kit_sets::kit14::sets_400bps_5khz}},
        {Chemistry::RNA002_70BPS,
         {"rna002_70bps", 3000, SampleType::RNA002, kit_sets::rna002::sets}},
        {Chemistry::RNA004_130BPS,
         {"rna004_130bps", 4000, SampleType::RNA004, kit_sets::rna004::sets}},
};

// Crate a map of flowcell and sequencing kit pairs to categorical chemistry
static ChemistryMap chemistry_map = [] {
    ChemistryMap map;
    for (const auto& [chemistry, kit_collection] : kit_map) {
        for (const auto& kset : kit_collection.kit_sets) {
            for (const auto& fc : kset.first) {
                for (const auto& kit : kset.second) {
                    map.insert({std::tuple(fc, kit, kit_collection.sampling_rate), chemistry});
                }
            }
        }
    }
    return map;
}();

}  // namespace chemistry

std::string to_string(const ChemistryKey& ck) {
    const auto [fc, kc, sr] = ck;
    return "flowcell_code: '" + to_string(fc) + "' sequencing_kit: '" + to_string(kc) +
           "' sample_rate: " + std::to_string(sr);
}

std::string to_string(const Chemistry& chemistry) {
    return get_info(chemistry, "chemistry", chemistry::kit_map).name;
}

const std::unordered_map<Chemistry, ChemistryKits>& chemistry_kits() { return chemistry::kit_map; }

const ChemistryMap& chemistry_map() { return chemistry::chemistry_map; }

Chemistry get_chemistry(const ChemistryKey& key) {
    const auto& map = chemistry_map();
    auto it = map.find(key);
    return it == map.end() ? Chemistry::UNKNOWN : it->second;
}

Chemistry get_chemistry(const std::string& chemistry) {
    return get_code(chemistry, Chemistry::UNKNOWN, chemistry_kits());
}

ChemistryKey get_chemistry_key(const std::string& flow_cell_product_code,
                               const std::string& sequencing_kit,
                               SamplingRate sample_rate) {
    const auto fc = models::flowcell_code(flow_cell_product_code);
    const auto kit = models::kit_code(sequencing_kit);
    const auto key = models::ChemistryKey(fc, kit, sample_rate);
    return key;
}

KitInfo ConditionInfo::get_kit_info() const { return kit_info(m_kit); };

const std::unordered_map<SampleType, SampleTypeInfo>& sample_types() {
    return chemistry::sample_type_map;
}

SampleType get_sample_type(const std::string& sample_type) {
    return get_code(sample_type, SampleType::UNKNOWN, sample_types());
}

SampleType get_sample_type_from_model_name(const std::string& model_name) {
    if (model_name.find("rna004") != std::string::npos) {
        return SampleType::RNA004;
    } else if (model_name.find("rna002") != std::string::npos) {
        return SampleType::RNA002;
    } else if (model_name.find("dna") != std::string::npos) {
        return SampleType::DNA;
    } else {
        return SampleType::UNKNOWN;
    }
}

SampleTypeInfo get_sample_type_info(const SampleType& sample_type) {
    return get_info(sample_type, "sample type", sample_types());
}

std::string to_string(const SampleType& sample_type) {
    return get_sample_type_info(sample_type).name;
}

}  // namespace dorado::models
