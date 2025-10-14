#pragma once

#include "metadata.h"
#include "models.h"

#include <optional>
#include <string>
#include <vector>

namespace dorado::models {

struct ModelComplex {
public:
    // Parse the model argument
    static ModelComplex parse(const std::string& raw);

    // Enumerated styles for how the user specified the models on the CLI
    enum class Style : uint8_t {
        PATH,
        NAMED,
        VARIANT,
    };

    Style style() const { return m_style; }
    bool is_variant_style() const { return m_style == Style::VARIANT; }
    bool is_named_style() const { return m_style == Style::NAMED; }
    bool is_path_style() const { return m_style == Style::PATH; }

    bool has_mods() const { return !m_mods_named.empty() || m_mod_variants.empty(); }

    const std::string& get_raw() const { return m_raw; }

    // Only valid IFF style::NAMED
    const ModelInfo& get_named_simplex_model() const;
    const std::vector<ModelInfo>& get_named_mods_models() const;

    // Only valid IFF style::VARIANT
    const ModelVariantPair& get_simplex_model_variant() const;
    const std::vector<ModsVariantPair>& get_mod_model_variants() const;

    bool operator==(const ModelComplex& other) const {
        return m_raw == other.m_raw && m_style == other.m_style;
    }
    bool operator!=(const ModelComplex& other) const { return !(*this == other); }

private:
    Style m_style{Style::PATH};  // The style of model complex used
    const std::string m_raw;     // Raw input string

    std::optional<ModelInfo> m_simplex_named;  // Simplex model info IFF style is NAME
    std::vector<ModelInfo> m_mods_named;       // Modbase model info IFF style is NAME

    std::optional<ModelVariantPair> m_simplex_variant;  // Simplex model info IFF style is VARIANT
    std::vector<ModsVariantPair> m_mod_variants;        // Modbase model info IFF style is VARIANT

protected:
    // ModelComplex constrcutor for NAME style
    ModelComplex(const std::string& raw,
                 const ModelInfo& simplex,
                 const std::vector<ModelInfo>& mods);
    // Parse the model argument from a model name string
    // "dna_r10.4.1_e8.2_400bps_sup@v5.2.0 -> (sup, v5.2.0)
    // "dna_r10.4.1_e8.2_400bps_sup@v5.2.0_5mC_5hmC@v2" -> (sup, v5.2.0), (sup@5mC_5hmC, v2.0.0)
    static std::optional<ModelComplex> parse_names(const std::string& raw);

    // ModelComplex constrcutor for VARIANT style
    ModelComplex(const std::string& raw,
                 const ModelVariantPair& simplex,
                 const std::vector<ModsVariantPair>& mods);
    // Parse the model argument from a variant string (e.g. hac,5mC@v2) -> (hac, latest), (hac@5mC, v2.0.0)
    static std::optional<ModelComplex> parse_variant(const std::string& raw);
    // Parse a single model variant part (e,g, fast@v4.2.0) -> (fast, v4.2.0)
    static std::pair<std::string, models::ModelVersion> parse_variant_part(const std::string& part);
    // Returns a well-formatted version string like "v0.0.0"
    static std::string parse_variant_version(const std::string& version);

    // ModelComplex constructor for PATH style;
    ModelComplex(const std::string& path);
};

}  // namespace dorado::models
