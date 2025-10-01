#pragma once

#include "kits.h"

#include <ostream>
#include <set>

namespace dorado::models {

/*
Enumerate the options available for auto model selection.
`UNRECOGNISED` is used to default back to a path 
*/
enum class ModelVariant : uint16_t {
    AUTO,
    FAST,
    HAC,
    SUP,
    NONE  // NONE must be last
};

// Enumeration of modifications codes
enum class ModsVariant : uint8_t {
    M_2OmeG,
    M_4mC_5mC,
    M_5mC_5hmC,
    M_5mCG,
    M_5mCG_5hmCG,
    M_5mC,
    M_m5C,
    M_m5C_2OmeC,
    M_6mA,
    M_m6A,
    M_m6A_DRACH,
    M_inosine_m6A,
    M_inosine_m6A_2OmeA,
    M_pseU,
    M_pseU_2OmeU,
    NONE  // NONE must be last
};

// version enumeration to ensure versions are easily sortable and type-safe
enum class ModelVersion : uint8_t {
    v1_0_0,
    v2_0_0,
    v2_0_1,
    v3_0_0,
    v3_0_1,
    v3_1_0,
    v3_5_2,
    v4_0_0,
    v4_1_0,
    v4_2_0,
    v4_3_0,
    v5_0_0,
    v5_1_0,
    v5_2_0,
    NONE  // NONE must be last
};

const std::unordered_map<std::string, ModelVariant>& model_variants_map();
const std::unordered_map<std::string, ModsVariant>& mods_variants_map();
const std::unordered_map<ModsVariant, std::string>& mods_canonical_base_map();
const std::unordered_map<ModsVariant, std::string>& mods_context_map();
const std::unordered_map<std::string, ModelVersion>& version_map();

ModelVariant get_model_variant(const std::string& variant);
ModsVariant get_mods_variant(const std::string& variant);
std::string get_mods_context(const ModsVariant& mod);

std::string to_string(const ModelVariant& variant);
std::string to_string(const ModsVariant& mod);
std::string to_string(const ModelVersion& version);
std::string to_string(const Chemistry& chemistry);

std::string to_string(const std::set<ModelVariant>& codes, const std::string& separator);

struct ModelVariantPair {
    ModelVariant variant = ModelVariant::NONE;
    ModelVersion ver = ModelVersion::NONE;
    bool is_auto = false;
    bool has_variant() const { return variant != ModelVariant::NONE; }
    bool has_ver() const { return ver != ModelVersion::NONE; }

    bool operator==(const ModelVariantPair& other) const {
        return other.variant == variant && other.ver == ver;
    }
};

inline std::ostream& operator<<(std::ostream& os, const ModelVariantPair& p) {
    os << "ModelVariantPair{" << to_string(p.variant) << ", " << to_string(p.ver) << '}';
    return os;
}

struct ModsVariantPair {
    ModsVariant variant = ModsVariant::NONE;
    ModelVersion ver = ModelVersion::NONE;
    bool has_variant() const { return variant != ModsVariant::NONE; }
    bool has_ver() const { return ver != ModelVersion::NONE; }

    bool operator==(const ModsVariantPair& other) const {
        return other.variant == variant && other.ver == ver;
    }
};

inline std::ostream& operator<<(std::ostream& os, const ModsVariantPair& p) {
    os << "ModsVariantPair{" << to_string(p.variant) << ", " << to_string(p.ver) << '}';
    return os;
}

}  // namespace dorado::models
