#pragma once

#include "models/kits.h"
#include "models/models.h"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace dorado {

struct ModelSelection {
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

    bool operator==(const ModelSelection& other) const { return raw == other.raw; }
    bool operator!=(const ModelSelection& other) const { return !(*this == other); }
};

class ModelComplexParser {
public:
    // Parse the model argument (e.g. hac,5mC@v2) -> (hac, latest), (hac@5mC, v2.0.0)
    static ModelSelection parse(const std::string& arg);

    // Given a potentially truncated version string, returns a well-formatted version
    // string like "v0.0.0" ensuring that there are 3 values, all values are integers,
    // and not empty
    static std::string parse_version(const std::string& version);

    // Parse a single model argument part (e,g, fast@v4.2.0) -> (fast, v4.2.0)
    static std::pair<std::string, models::ModelVersion> parse_model_arg_part(
            const std::string& part);
};

class ModelFinder {
public:
    ModelFinder(const models::Chemistry& chemsitry,
                const ModelSelection& selection,
                bool suggestions);

    // Return the selection
    ModelSelection get_selection() { return m_selection; }
    // Return the chemistry found
    models::Chemistry get_chemistry() { return m_chemistry; }

    //Find a simplex model which matches the user's command and chemistry
    std::string get_simplex_model_name() const;
    //Find a stereo model which matches the chemistry
    std::string get_stereo_model_name() const;
    // Find modification models which match the user's mods selections and chemistry
    std::vector<std::string> get_mods_model_names() const;
    // Find modification models which matches the simplex model and chemistry
    std::vector<std::string> get_mods_for_simplex_model() const;

    // Get the simplex model (auto-download) and return the path
    std::filesystem::path fetch_simplex_model();
    // Get the stereo model (auto-download) and return the path
    std::filesystem::path fetch_stereo_model();
    // Get the mods models (auto-download) and return the paths
    std::vector<std::filesystem::path> fetch_mods_models();

    std::set<std::filesystem::path> downloaded_models() const { return m_downloaded_models; };

    // Inspects the sequencing data metadata to determine the sequencing chemistry used. Will error if
    // the data is inhomogeneous
    static models::Chemistry inspect_chemistry(const std::string& data,
                                               bool recursive_file_loading);

    // Search simpex models by name, throws exceptions if not found
    static models::ModelInfo get_simplex_model_info(const std::string& model_name);

private:
    // If a ModelVariant was set, the chemistry (e.g. R10.4.1 / RNA004) is deduced from the
    // sequencing data otherwise it is meaningless
    const models::Chemistry m_chemistry;
    const ModelSelection m_selection;
    // If true, show suggestions if no models were found given some settings.
    const bool m_suggestions;
    // Store the result of the simplex model call to resolve mods when ModelVariant::AUTO is used
    const models::ModelInfo m_simplex_model_info;
    // Returns the model path after possibly downloading the model if it wasn't found
    // in the expected locations
    std::filesystem::path fetch_model(const std::string& model_name,
                                      const std::string& description);

    models::ModelInfo get_simplex_model_info() const;

    // Set of downloaded models which we want to clean up on shutdown.
    std::set<std::filesystem::path> m_downloaded_models;
};

// Attempts to assert that the model sampling rate and data sampling rate are compatible.
// Warns if testing fails. Throws an exception if sampling rates are known to be incompatible.
void check_sampling_rates_compatible(const std::string& model_name,
                                     const std::filesystem::path& data_path,
                                     const int config_sample_rate,
                                     const bool recursive_file_loading);

// Get modified models set using `--modified-bases` or `--modified-bases-models` cli args
std::vector<std::filesystem::path> get_non_complex_mods_models(
        const std::filesystem::path& simplex_model_path,
        const std::vector<std::string>& mod_bases,
        const std::string& mod_bases_models);

}  // namespace dorado