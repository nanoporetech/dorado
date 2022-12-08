#include "models.h"

#include <elzip/elzip.hpp>

#include <filesystem>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace dorado::utils {

bool is_valid_model(const std::string& selected_model) {
    return selected_model == "all" ||
           urls::simplex::models.find(selected_model) != urls::simplex::models.end() ||
           urls::stereo::models.find(selected_model) != urls::stereo::models.end() ||
           urls::modified::models.find(selected_model) != urls::modified::models.end();
}

void download_models(const std::string& target_directory, const std::string& selected_model) {
    fs::path directory(target_directory);

    httplib::Client http(urls::URL_ROOT);
    http.enable_server_certificate_verification(false);
    http.set_follow_location(true);

    auto download_model_set = [&](std::map<std::string, std::string> models) {
        for (const auto& [model, url] : models) {
            if (selected_model == "all" || selected_model == model) {
                spdlog::info(" - downloading {}", model);
                auto res = http.Get(url.c_str());
                if (res != nullptr) {
                    fs::path archive(directory / (model + ".zip"));
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

    download_model_set(urls::simplex::models);
    download_model_set(urls::stereo::models);
    download_model_set(urls::modified::models);
}

bool is_rna_model(const std::filesystem::path& model) {
    auto path = fs::canonical(model);
    auto filename = path.filename();
    return filename.u8string().rfind("rna", 0) == 0;
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
        for (const auto& [model, _] : urls::modified::models) {
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

}  // namespace dorado::utils
