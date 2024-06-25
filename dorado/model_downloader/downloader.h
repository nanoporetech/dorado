#pragma once

#include "models/models.h"

#include <filesystem>
#include <memory>

namespace httplib {
class Client;
}  // namespace httplib

namespace dorado::model_downloader {

class Downloader {
public:
    Downloader(std::filesystem::path directory);
    ~Downloader();

    bool download(const models::ModelInfo& info);

private:
#if DORADO_MODELS_HAS_HTTPLIB
    std::unique_ptr<httplib::Client> m_client;
#endif  // DORADO_MODELS_HAS_HTTPLIB
    const std::filesystem::path m_directory;

    std::string get_url(const std::string& model) const;
    bool validate_checksum(std::string_view data, const models::ModelInfo& model) const;
    void extract(const std::filesystem::path& archive) const;

#if DORADO_MODELS_HAS_HTTPLIB
    static auto create_client();
    bool download_httplib(const models::ModelInfo& model, const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_HTTPLIB
#if DORADO_MODELS_HAS_CURL_EXE
    bool download_curl(const models::ModelInfo& model, const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_CURL_EXE
#if DORADO_MODELS_HAS_FOUNDATION
    bool download_foundation(const models::ModelInfo& model, const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_FOUNDATION
};

}  // namespace dorado::model_downloader
