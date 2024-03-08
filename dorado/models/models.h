#pragma once

#include "kits.h"
#include "metadata.h"

#include <filesystem>
#include <optional>

namespace dorado::models {

// Model info identifies models and associates additional metadata for searching for
// automatic model selection.
struct ModelInfo {
    std::string name;
    std::string checksum;
    Chemistry chemistry;
    ModelVariantPair simplex{};
    ModsVariantPair mods{};
};

// Search for a model which is configured for use with the given chemistry and other
// optional params.
ModelInfo find_model(const std::vector<ModelInfo>& models,
                     const std::string& description,
                     const Chemistry& chemistry,
                     const ModelVariantPair& model,
                     const ModsVariantPair& mods,
                     bool suggestions);

// Search for models which match the given chemistry and filters. Returns all matches
// in ascending version order
std::vector<ModelInfo> find_models(const std::vector<ModelInfo>& models,
                                   const Chemistry& chemistry,
                                   const ModelVariantPair& model,
                                   const ModsVariantPair& mods);

using ModelList = std::vector<ModelInfo>;
const ModelList& simplex_models();
const ModelList& stereo_models();
const ModelList& modified_models();

std::vector<std::string> simplex_model_names();
std::vector<std::string> stereo_model_names();
std::vector<std::string> modified_model_names();
std::vector<std::string> modified_model_variants();

bool is_valid_model(const std::string& selected_model);
bool download_models(const std::string& target_directory, const std::string& selected_model);

// Search for a simplex model by name and return the ModelInfo
ModelInfo get_simplex_model_info(const std::string& model_name);

// finds the matching modification model for a given modification i.e. 5mCG and a simplex model
// if the modification model is not found in the same model directory as the simplex
// model then it is downloaded into the same directory.
std::string get_modification_model(const std::filesystem::path& simplex_model,
                                   const std::string& modification);

// get the sampling rate that the model is compatible with
SamplingRate get_sample_rate_by_model_name(const std::string& model_name);

// Extract the model name from the model path.
std::string extract_model_name_from_path(const std::filesystem::path& model_path);

// Extract the model names as a comma seperated list from a vetor of model paths.
std::string extract_model_names_from_paths(const std::vector<std::filesystem::path>& model_paths);

}  // namespace dorado::models
