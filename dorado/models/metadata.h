#pragma once

#include "kits.h"

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
    M_4mC_5mC,
    M_5mC_5hmC,
    M_5mCG,
    M_5mCG_5hmCG,
    M_5mC,
    M_m5C,
    M_6mA,
    M_m6A,
    M_m6A_DRACH,
    M_inosine_m6A,
    M_pseU,
    NONE  // NONE must be last
};

// version enumeration to ensure versions are easily sortable and type-safe
enum class ModelVersion : uint8_t {
    v0_0_0,
    v0_1_0,
    v1_0_0,
    v1_1_0,
    v1_2_0,
    v2_0_0,
    v2_0_1,
    v3_0_0,
    v3_0_1,
    v3_1_0,
    v3_3_0,
    v3_4_0,
    v3_5_0,
    v3_5_2,
    v3_6_0,
    v4_0_0,
    v4_1_0,
    v4_2_0,
    v4_3_0,
    v5_0_0,
    v5_1_0,
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

struct ModsVariantPair {
    ModsVariant variant = ModsVariant::NONE;
    ModelVersion ver = ModelVersion::NONE;
    bool has_variant() const { return variant != ModsVariant::NONE; }
    bool has_ver() const { return ver != ModelVersion::NONE; }
};

}  // namespace dorado::models
