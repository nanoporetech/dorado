#include "models.h"

#include <elzip/elzip.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

namespace dorado::utils {

namespace {

namespace urls {

const std::string URL_ROOT = "https://cdn.oxfordnanoportal.com";
const std::string URL_PATH = "/software/analysis/dorado/";

}  // namespace urls

// Serialised, released models
namespace simplex {

const ModelMap models = {

        // v3.{3,4,6}
        {"dna_r9.4.1_e8_fast@v3.4", {""}},
        {"dna_r9.4.1_e8_hac@v3.3", {""}},
        {"dna_r9.4.1_e8_sup@v3.3", {""}},
        {"dna_r9.4.1_e8_sup@v3.6", {""}},

        // v3.5.2
        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2", {""}},

        // v4.0.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.0.0", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.0.0", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.0.0", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0", {""}},

        // v4.1.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.1.0", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.1.0", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.1.0", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.1.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.1.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.1.0", {""}},

        // v4.2.0
        {"dna_r10.4.1_e8.2_400bps_fast@v4.2.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0", {""}},

        // RNA002
        {"rna002_70bps_fast@v3", {""}},
        {"rna002_70bps_hac@v3", {""}},

        // RNA004
        {"rna004_130bps_fast@v3.0.1", {""}},
        {"rna004_130bps_hac@v3.0.1", {""}},
        {"rna004_130bps_sup@v3.0.1", {""}},
};

}  // namespace simplex

namespace stereo {

const ModelMap models = {
        {"dna_r10.4.1_e8.2_4khz_stereo@v1.1", {""}},
        {"dna_r10.4.1_e8.2_5khz_stereo@v1.1", {""}},
};

}  // namespace stereo

namespace modified {

const std::vector<std::string> mods = {
        "5mCG",
        "5mCG_5hmCG",
        "5mC",
        "6mA",
};

const ModelMap models = {

        // v3.{3,4}
        {"dna_r9.4.1_e8_fast@v3.4_5mCG@v0.1", {""}},
        {"dna_r9.4.1_e8_hac@v3.3_5mCG@v0.1", {""}},
        {"dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1", {""}},

        {"dna_r9.4.1_e8_fast@v3.4_5mCG_5hmCG@v0", {""}},
        {"dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0", {""}},
        {"dna_r9.4.1_e8_sup@v3.3_5mCG_5hmCG@v0", {""}},

        // v3.5.2
        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2", {""}},

        // v4.0.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.0.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.0.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2", {""}},

        // v4.1.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.1.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.1.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2", {""}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.1.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.1.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.1.0_5mCG_5hmCG@v2", {""}},

        // v4.2.0
        {"dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2", {""}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2", {""}},

};

}  // namespace modified

const std::unordered_map<std::string, uint16_t> sample_rate_by_model = {

        //------ simplex ---------//
        // v4.2
        {"dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0", 5000},
        {"dna_r10.4.1_e8.2_5khz_400bps_hac@v4.2.0", 5000},
        {"dna_r10.4.1_e8.2_5khz_400bps_sup@v4.2.0", 5000},

        //------ duplex ---------//
        // v4.2
        {"dna_r10.4.1_e8.2_5khz_stereo@v1.1", 5000},
};

const std::unordered_map<std::string, uint16_t> mean_qscore_start_pos_by_model = {

        // To add model specific start positions for older models,
        // create an entry keyed by model name with the value as
        // the desired start position.
        // e.g. {"dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0", 10}
};

}  // namespace

const ModelMap& simplex_models() { return simplex::models; }
const ModelMap& stereo_models() { return stereo::models; }
const ModelMap& modified_models() { return modified::models; }
const std::vector<std::string>& modified_mods() { return modified::mods; }

bool is_valid_model(const std::string& selected_model) {
    return selected_model == "all" || simplex::models.count(selected_model) > 0 ||
           stereo::models.count(selected_model) > 0 || modified::models.count(selected_model) > 0;
}

void download_models(const std::string& target_directory, const std::string& selected_model) {
    fs::path directory(target_directory);

    httplib::Client http(urls::URL_ROOT);
    http.enable_server_certificate_verification(false);
    http.set_follow_location(true);

    const char* proxy_url = getenv("dorado_proxy");
    const char* ps = getenv("dorado_proxy_port");

    int proxy_port = 3128;
    if (ps) {
        proxy_port = atoi(ps);
    }

    if (proxy_url) {
        spdlog::info("using proxy: {}:{}", proxy_url, proxy_port);
        http.set_proxy(proxy_url, proxy_port);
    }

    auto download_model_set = [&](const ModelMap& models) {
        for (const auto& [model, info] : models) {
            if (selected_model == "all" || selected_model == model) {
                // TIL operator+ doesn't exist for string and string_view -_-
                const std::string model_str(model);
                auto url = urls::URL_ROOT + urls::URL_PATH + model_str + ".zip";
                spdlog::info(" - downloading {}", model);
                auto res = http.Get(url.c_str());
                if (res != nullptr) {
                    fs::path archive(directory / (model_str + ".zip"));
                    std::ofstream ofs(archive.string(), std::ofstream::binary);
                    ofs << res->body;
                    ofs.close();
                    elz::extractZip(archive, directory);
                    fs::remove(archive);
                } else {
                    spdlog::error("Failed to download {}", model);
                }
            }
        }
    };

    download_model_set(simplex::models);
    download_model_set(stereo::models);
    download_model_set(modified::models);
}

std::string get_modification_model(const std::string& simplex_model,
                                   const std::string& modification) {
    std::string modification_model{""};
    auto simplex_path = fs::path(simplex_model);

    if (!fs::exists(simplex_path)) {
        throw std::runtime_error{"unknown simplex model " + simplex_model};
    }

    simplex_path = fs::canonical(simplex_path);
    auto model_dir = simplex_path.parent_path();
    auto simplex_name = simplex_path.filename().u8string();

    if (is_valid_model(simplex_name)) {
        std::string mods_prefix = simplex_name + "_" + modification + "@v";
        for (const auto& [model, info] : modified::models) {
            if (model.compare(0, mods_prefix.size(), mods_prefix) == 0) {
                modification_model = model;
                break;
            }
        }
    } else {
        throw std::runtime_error{"unknown simplex model " + simplex_name};
    }

    if (modification_model.empty()) {
        throw std::runtime_error{"could not find matching modification model for " + simplex_name};
    }

    spdlog::debug("- matching modification model found: {}", modification_model);

    auto modification_path = model_dir / fs::path{modification_model};
    if (!fs::exists(modification_path)) {
        download_models(model_dir.u8string(), modification_model);
    }

    return modification_path.u8string();
}

uint16_t get_sample_rate_by_model_name(const std::string& model_name) {
    auto iter = sample_rate_by_model.find(model_name);
    if (iter != sample_rate_by_model.end()) {
        return iter->second;
    } else {
        // Assume any model not found in the list has sample rate 4000.
        return 4000;
    }
}

uint32_t get_mean_qscore_start_pos_by_model_name(const std::string& model_name) {
    auto iter = mean_qscore_start_pos_by_model.find(model_name);
    if (iter != mean_qscore_start_pos_by_model.end()) {
        return iter->second;
    } else {
        // Assume start position of 60 as default.
        return 60;
    }
}

std::string extract_model_from_model_path(const std::string& model_path) {
    std::filesystem::path path(model_path);
    return std::filesystem::canonical(path).filename().string();
}

}  // namespace dorado::utils
