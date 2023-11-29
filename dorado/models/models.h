#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::models {

struct ModelInfo {
    std::string_view checksum;
};
using ModelMap = std::map<std::string_view, ModelInfo>;

const ModelMap& simplex_models();
const ModelMap& stereo_models();
const ModelMap& modified_models();
const std::vector<std::string>& modified_mods();

bool is_valid_model(const std::string& selected_model);
bool download_models(const std::string& target_directory, const std::string& selected_model);

// finds the matching modification model for a given modification i.e. 5mCG and a simplex model
// is the matching modification model is not found in the same model directory as the simplex
// model then it is downloaded.
std::string get_modification_model(const std::string& simplex_model,
                                   const std::string& modification);

// fetch the sampling rate that the model is compatible with. for models not
// present in the mapping, assume a sampling rate of 4000.
uint16_t get_sample_rate_by_model_name(const std::string& model_name);

// the mean Q-score of short reads are artificially lowered because of
// some lower quality bases at the beginning of the read. to correct for
// that, mean Q-score calculation should ignore the first few bases. The
// number of bases to ignore is dependent on the model.
uint32_t get_mean_qscore_start_pos_by_model_name(const std::string& model_name);

// Extract the model name from the model path.
std::string extract_model_from_model_path(const std::string& model_path);

// Extract the model names as a comma seperated list from a comma seperated list of model paths.
std::string extract_model_from_model_paths(const std::string& model_paths);

}  // namespace dorado::models
