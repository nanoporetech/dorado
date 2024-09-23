#pragma once

#include "metadata.h"
#include "models.h"

#include <string>
#include <vector>

namespace dorado::models {

struct ModelComplex {
    std::string raw;
    models::ModelVariantPair model =
            models::ModelVariantPair{models::ModelVariant::NONE, models::ModelVersion::NONE};
    std::vector<models::ModsVariantPair> mods = {};

    // Returns true of the model argument was a command option (e.g. auto / fast / hac / sup)
    bool has_model_variant() const { return model.variant != models::ModelVariant::NONE; }
    // Returns true if the model argument appears to be a path
    bool is_path() const { return !has_model_variant(); }
    // Returns true if mods models were specified in the model command
    bool has_mods_variant() const { return mods.size() > 0; }

    bool operator==(const ModelComplex& other) const { return raw == other.raw; }
    bool operator!=(const ModelComplex& other) const { return !(*this == other); }
};

class ModelComplexParser {
public:
    // Parse the model argument (e.g. hac,5mC@v2) -> (hac, latest), (hac@5mC, v2.0.0)
    static ModelComplex parse(const std::string& arg);

    // Given a potentially truncated version string, returns a well-formatted version
    // string like "v0.0.0" ensuring that there are 3 values, all values are integers,
    // and not empty
    static std::string parse_version(const std::string& version);

    // Parse a single model argument part (e,g, fast@v4.2.0) -> (fast, v4.2.0)
    static std::pair<std::string, models::ModelVersion> parse_model_arg_part(
            const std::string& part);
};

// The ModelComplexSearch takes the sequencing chemistry, and the user's ModelComplex input which
// could be a actual model complex tries to find the ModelInfo from the models lib.
class ModelComplexSearch {
public:
    ModelComplexSearch(const ModelComplex& selection, Chemistry chemistry, bool suggestions);
    // Return the model complex
    ModelComplex complex() { return m_complex; }
    // Return the chemistry
    Chemistry chemistry() { return m_chemistry; }
    // Return the simplex model found during initialisation
    ModelInfo simplex() const;
    // Find a stereo model which matches the chemistry
    ModelInfo stereo() const;
    // Find modification models which match the user's mods selections and chemistry
    std::vector<ModelInfo> mods() const;
    // Find all modification models which matches the simplex model and chemistry
    std::vector<ModelInfo> simplex_mods() const;

private:
    // Resolve the simplex model which matches the user's command and chemistry
    ModelInfo resolve_simplex() const;
    // The user's model complex input
    const ModelComplex m_complex;
    // If a ModelVariant was set, the chemistry (e.g. R10.4.1 / RNA004) is deduced from the
    // sequencing data otherwise it is meaningless
    const Chemistry m_chemistry;
    // If true, show suggestions if no models were found given some settings.
    const bool m_suggestions;
    // Store the result of the simplex model search
    const ModelInfo m_simplex_model_info;
};

}  // namespace dorado::models
