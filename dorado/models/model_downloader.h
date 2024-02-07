#pragma once

#include <filesystem>
#include <memory>
#include <string>

namespace httplib {
class Client;
}  // namespace httplib

namespace dorado::models {
struct ModelInfo;

class ModelDownloader {
public:
    ModelDownloader(std::filesystem::path directory);
    ~ModelDownloader();

    bool download(const std::string& model, const ModelInfo& info);

private:
#if DORADO_MODELS_HAS_HTTPLIB
    std::unique_ptr<httplib::Client> m_client;
#endif  // DORADO_MODELS_HAS_HTTPLIB
    const std::filesystem::path m_directory;

    std::string get_url(const std::string& model) const;
    bool validate_checksum(std::string_view data, const ModelInfo& info) const;
    void extract(const std::filesystem::path& archive) const;

#if DORADO_MODELS_HAS_HTTPLIB
    static auto create_client();
    bool download_httplib(const std::string& model,
                          const ModelInfo& info,
                          const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_HTTPLIB
#if DORADO_MODELS_HAS_CURL_EXE
    bool download_curl(const std::string& model,
                       const ModelInfo& info,
                       const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_CURL_EXE
#if DORADO_MODELS_HAS_FOUNDATION
    bool download_foundation(const std::string& model,
                             const ModelInfo& info,
                             const std::filesystem::path& archive);
#endif  // DORADO_MODELS_HAS_FOUNDATION
};

}  // namespace dorado::models
