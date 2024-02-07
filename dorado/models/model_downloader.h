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
    std::unique_ptr<httplib::Client> m_client;
    const std::filesystem::path m_directory;

    std::string get_url(const std::string& model) const;
    bool validate_checksum(std::string_view data, const ModelInfo& info) const;
    void extract(const std::filesystem::path& archive) const;

    bool download_httplib(const std::string& model,
                          const ModelInfo& info,
                          const std::filesystem::path& archive);
    bool download_curl(const std::string& model,
                       const ModelInfo& info,
                       const std::filesystem::path& archive);
};

}  // namespace dorado::models
