#include <filesystem>
#include <string>
#include <vector>

namespace dorado {
namespace urls {

static const std::string URL_ROOT = "https://cdn.oxfordnanoportal.com";
static const std::string URL_PATH = "/software/analysis/dorado/";

}  // namespace urls

// Serialised, released models
namespace simplex {

static const std::vector<std::string> models = {

        // v3.{3,4}
        "dna_r9.4.1_e8_fast@v3.4",
        "dna_r9.4.1_e8_hac@v3.3",
        "dna_r9.4.1_e8_sup@v3.3",

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

        // RNA003
        "rna003_120bps_sup@v3",
};

}  // namespace simplex

namespace stereo {

static const std::vector<std::string> models = {"dna_r10.4.1_e8.2_4khz_stereo@v1.1"};

}

namespace modified {

static const std::vector<std::string> mods = {
        "5mCG",
        "5mCG_5hmCG",
};

static const std::vector<std::string> models = {

        // v3.{3,4}
        "dna_r9.4.1_e8_fast@v3.4_5mCG@v0",
        "dna_r9.4.1_e8_hac@v3.4_5mCG@v0",
        "dna_r9.4.1_e8_sup@v3.4_5mCG@v0",

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

};

}  // namespace modified

namespace utils {

bool is_rna_model(const std::filesystem::path& model);
bool is_valid_model(const std::string& selected_model);
void download_models(const std::string& target_directory, const std::string& selected_model);

// finds the matching modification model for a given modification i.e. 5mCG and a simplex model
// is the matching modification model is not found in the same model directory as the simplex
// model then it is downloaded.
std::string get_modification_model(const std::string& simplex_model,
                                   const std::string& modification);

}  // namespace utils

}  // namespace dorado
