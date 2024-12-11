#include "metadata.h"

#include "kits.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace dorado::models {

namespace model_variant {
const std::unordered_map<std::string, ModelVariant> map = {
        {"auto", ModelVariant::AUTO},
        {"fast", ModelVariant::FAST},
        {"hac", ModelVariant::HAC},
        {"sup", ModelVariant::SUP},
};
}

namespace mods {
const std::unordered_map<std::string, ModsVariant> map = {
        {"4mC_5mC", ModsVariant::M_4mC_5mC},
        {"5mC_5hmC", ModsVariant::M_5mC_5hmC},
        {"5mCG", ModsVariant::M_5mCG},
        {"5mCG_5hmCG", ModsVariant::M_5mCG_5hmCG},
        {"5mC", ModsVariant::M_5mC},
        {"m5C", ModsVariant::M_m5C},
        {"6mA", ModsVariant::M_6mA},
        {"m6A", ModsVariant::M_m6A},
        {"m6A_DRACH", ModsVariant::M_m6A_DRACH},
        {"inosine_m6A", ModsVariant::M_inosine_m6A},
        {"pseU", ModsVariant::M_pseU},
};

const std::unordered_map<ModsVariant, std::string> canonical_base_map = {
        {ModsVariant::M_4mC_5mC, "C"}, {ModsVariant::M_5mC_5hmC, "C"},
        {ModsVariant::M_5mCG, "C"},    {ModsVariant::M_5mCG_5hmCG, "C"},
        {ModsVariant::M_5mC, "C"},     {ModsVariant::M_m5C, "C"},
        {ModsVariant::M_6mA, "A"},     {ModsVariant::M_inosine_m6A, "A"},
        {ModsVariant::M_m6A, "A"},     {ModsVariant::M_m6A_DRACH, "A"},
        {ModsVariant::M_pseU, "T"},
};

const std::unordered_map<ModsVariant, std::string> context_map = {
        {ModsVariant::M_4mC_5mC, "C"}, {ModsVariant::M_5mC_5hmC, "C"},
        {ModsVariant::M_5mCG, "CG"},   {ModsVariant::M_5mCG_5hmCG, "CG"},
        {ModsVariant::M_5mC, "C"},     {ModsVariant::M_m5C, "C"},
        {ModsVariant::M_6mA, "A"},     {ModsVariant::M_inosine_m6A, "A"},
        {ModsVariant::M_m6A, "A"},     {ModsVariant::M_m6A_DRACH, "DRACH"},
        {ModsVariant::M_pseU, "T"},
};

}  // namespace mods

namespace version {
const std::unordered_map<std::string, ModelVersion> map = {
        {"v0.0.0", ModelVersion::v0_0_0}, {"v0.1.0", ModelVersion::v0_1_0},
        {"v1.0.0", ModelVersion::v1_0_0}, {"v1.1.0", ModelVersion::v1_1_0},
        {"v1.2.0", ModelVersion::v1_2_0}, {"v2.0.0", ModelVersion::v2_0_0},
        {"v2.0.1", ModelVersion::v2_0_1}, {"v3.0.0", ModelVersion::v3_0_0},
        {"v3.0.1", ModelVersion::v3_0_1}, {"v3.1.0", ModelVersion::v3_1_0},
        {"v3.3.0", ModelVersion::v3_3_0}, {"v3.4.0", ModelVersion::v3_4_0},
        {"v3.5.0", ModelVersion::v3_5_0}, {"v3.5.2", ModelVersion::v3_5_2},
        {"v3.6.0", ModelVersion::v3_6_0}, {"v4.0.0", ModelVersion::v4_0_0},
        {"v4.1.0", ModelVersion::v4_1_0}, {"v4.2.0", ModelVersion::v4_2_0},
        {"v4.3.0", ModelVersion::v4_3_0}, {"v5.0.0", ModelVersion::v5_0_0},
        {"v5.1.0", ModelVersion::v5_1_0}, {"latest", ModelVersion::NONE}};
}  // namespace version

const std::unordered_map<std::string, ModelVariant>& model_variants_map() {
    return model_variant::map;
}
const std::unordered_map<std::string, ModsVariant>& mods_variants_map() { return mods::map; }
const std::unordered_map<ModsVariant, std::string>& mods_canonical_base_map() {
    return mods::canonical_base_map;
}
const std::unordered_map<ModsVariant, std::string>& mods_context_map() { return mods::context_map; }
const std::unordered_map<std::string, ModelVersion>& version_map() { return version::map; }

namespace {

template <typename Variant>
Variant get_variant(const std::string& variant,
                    const std::unordered_map<std::string, Variant>& variants) {
    auto it = variants.find(variant);
    if (it == std::end(variants)) {
        return Variant::NONE;
    }
    return it->second;
}

template <typename Variant>
std::string to_string(const Variant& variant,
                      const std::string& description,
                      const std::unordered_map<std::string, Variant>& variants) {
    auto it = std::find_if(std::begin(variants), std::end(variants),
                           [&variant](auto&& kv) { return kv.second == variant; });

    if (it == std::end(variants)) {
        throw std::logic_error("Unknown + " + description +
                               " enum: " + std::to_string(static_cast<int>(variant)));
    }
    return it->first;
}

}  // namespace

ModelVariant get_model_variant(const std::string& variant) {
    return get_variant(variant, model_variants_map());
}

ModsVariant get_mods_variant(const std::string& variant) {
    return get_variant(variant, mods_variants_map());
}

std::string get_mods_context(const ModsVariant& variant) {
    const auto& contexts = mods_context_map();
    auto it = contexts.find(variant);
    if (it == std::end(contexts)) {
        const std::string var_str = to_string(variant);
        throw std::logic_error("Unknown modification in context mapping: " + var_str);
    }
    return it->second;
}

std::string to_string(const ModelVariant& variant) {
    return to_string(variant, "model variant", model_variants_map());
}

std::string to_string(const ModsVariant& variant) {
    return to_string(variant, "modification variant", mods_variants_map());
}

std::string to_string(const std::set<ModelVariant>& variants, const std::string& separator) {
    std::vector<std::string> strings;
    strings.reserve(variants.size());
    for (const auto& variant : variants) {
        strings.push_back(to_string(variant, "model variant", model_variants_map()));
    }
    return utils::join(strings, separator);
}

std::string to_string(const ModelVersion& version) {
    return to_string(version, "model version", version_map());
}

}  // namespace dorado::models
