#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace dorado {
namespace urls {

static const std::string URL_ROOT = "https://cdn.oxfordnanoportal.com";
static const std::string URL_PATH = "/software/analysis/dorado/";

}  // namespace urls

// Serialised, released models
namespace simplex {

static const std::vector<std::string> models = {

        // v3.{3,4,6}
        "dna_r9.4.1_e8_fast@v3.4",
        "dna_r9.4.1_e8_hac@v3.3",
        "dna_r9.4.1_e8_sup@v3.3",
        "dna_r9.4.1_e8_sup@v3.6",

        // v3.5.2
        "dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
        "dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
        "dna_r10.4.1_e8.2_260bps_sup@v3.5.2",

        "dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
        "dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
        "dna_r10.4.1_e8.2_400bps_sup@v3.5.2",

        // v4.0.0
        "dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
        "dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
        "dna_r10.4.1_e8.2_260bps_sup@v4.0.0",

        "dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
        "dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
        "dna_r10.4.1_e8.2_400bps_sup@v4.0.0",

        // v4.1.0
        "dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
        "dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
        "dna_r10.4.1_e8.2_260bps_sup@v4.1.0",

        "dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
        "dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
        "dna_r10.4.1_e8.2_400bps_sup@v4.1.0",

        // v4.2.0
        "dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
        "dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
        "dna_r10.4.1_e8.2_400bps_sup@v4.2.0",

        // RNA002
        "rna002_70bps_fast@v3",
        "rna002_70bps_hac@v3",

        // RNA003
        "rna003_120bps_sup@v3",

        // RNA004
        "rna004_130bps_fast@v3",
        "rna004_130bps_hac@v3",
        "rna004_130bps_sup@v3",
};

}  // namespace simplex

namespace stereo {

static const std::vector<std::string> models = {
        "dna_r10.4.1_e8.2_4khz_stereo@v1.1",
        "dna_r10.4.1_e8.2_5khz_stereo@v1.1",
};

}  // namespace stereo

namespace modified {

static const std::vector<std::string> mods = {
        "5mCG",
        "5mCG_5hmCG",
        "5mC",
        "6mA",
};

static const std::vector<std::string> models = {

        // v3.{3,4}
        "dna_r9.4.1_e8_fast@v3.4_5mCG@v0",
        "dna_r9.4.1_e8_hac@v3.3_5mCG@v0",
        "dna_r9.4.1_e8_sup@v3.3_5mCG@v0",

        "dna_r9.4.1_e8_fast@v3.4_5mCG_5hmCG@v0",
        "dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0",
        "dna_r9.4.1_e8_sup@v3.3_5mCG_5hmCG@v0",

        // v3.5.2
        "dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2",
        "dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2",
        "dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2",

        "dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2",
        "dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2",

        // v4.0.0
        "dna_r10.4.1_e8.2_260bps_fast@v4.0.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_260bps_hac@v4.0.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2",

        "dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2",

        // v4.1.0
        "dna_r10.4.1_e8.2_260bps_fast@v4.1.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_260bps_hac@v4.1.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2",

        "dna_r10.4.1_e8.2_400bps_fast@v4.1.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_hac@v4.1.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v4.1.0_5mCG_5hmCG@v2",

        // v4.2.0
        "dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_hac@v4.2.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2",
        "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2",

};

}  // namespace modified

namespace utils {

static const std::unordered_map<std::string, uint16_t> sample_rate_by_model = {

        //------ simplex ---------//
        // v4.2
        {"dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0", 5000},
        {"dna_r10.4.1_e8.2_5khz_400bps_hac@v4.2.0", 5000},
        {"dna_r10.4.1_e8.2_5khz_400bps_sup@v4.2.0", 5000},

        //------ duplex ---------//
        // v4.2
        {"dna_r10.4.1_e8.2_5khz_stereo@v1.1", 5000},
};

static const std::unordered_map<std::string, uint16_t> mean_qscore_start_pos_by_model = {

        // To add model specific start positions for older models,
        // create an entry keyed by model name with the value as
        // the desired start position.
        // e.g. {"dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0", 10}
};

bool is_rna_model(const std::filesystem::path& model);
bool is_valid_model(const std::string& selected_model);
void download_models(const std::string& target_directory, const std::string& selected_model);

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

}  // namespace utils

}  // namespace dorado
