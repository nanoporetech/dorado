#include "models.h"

#include <elzip/elzip.hpp>

#include <filesystem>
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

namespace dorado::utils {

void download_models(std::string target_directory, std::string selected_model) {
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
    download_model_set(urls::modified::models);
}

}  // namespace dorado::utils