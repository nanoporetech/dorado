#include "downloader.h"

#include "models/models.h"
#include "utils/crypto_utils.h"

#include <elzip/elzip.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <sstream>

#if DORADO_MODELS_HAS_HTTPLIB
#ifndef _WIN32
// Required for MSG_NOSIGNAL and SO_NOSIGPIPE
#include <sys/socket.h>
#include <sys/types.h>
#endif

#ifdef MSG_NOSIGNAL
#define CPPHTTPLIB_SEND_FLAGS MSG_NOSIGNAL
#endif
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#endif  // DORADO_MODELS_HAS_HTTPLIB

namespace fs = std::filesystem;

namespace dorado::model_downloader {

namespace {

namespace urls {
const std::string URL_ROOT = "https://cdn.oxfordnanoportal.com";
const std::string URL_PATH = "/software/analysis/dorado/";
}  // namespace urls

std::string calculate_checksum(std::string_view data) {
    // Hash the data.
    const auto hash = utils::crypto::sha256(data);

    // Stringify it.
    std::ostringstream checksum;
    checksum << std::hex;
    checksum.fill('0');
    for (unsigned char byte : hash) {
        checksum << std::setw(2) << static_cast<int>(byte);
    }
    return std::move(checksum).str();
}

#if DORADO_MODELS_HAS_HTTPLIB
void set_ssl_cert_file() {
#ifndef _WIN32
    // Allow the user to override this.
    if (getenv("SSL_CERT_FILE") != nullptr) {
        return;
    }

    // Try and find the cert location.
    const char* ssl_cert_file = nullptr;
#ifdef __linux__
    // We link to a static Ubuntu build of OpenSSL so it's expecting certs to be where Ubuntu puts them.
    // For other distributions they may not be in the same place or have the same name.
    if (fs::exists("/etc/os-release")) {
        std::ifstream os_release("/etc/os-release");
        std::string line;
        while (std::getline(os_release, line)) {
            if (line.rfind("ID=", 0) == 0) {
                if (line.find("ubuntu") != line.npos || line.find("debian") != line.npos) {
                    // SSL will pick the right one.
                    return;
                } else if (line.find("centos") != line.npos) {
                    ssl_cert_file = "/etc/ssl/certs/ca-bundle.crt";
                }
                break;
            }
        }
    }
    if (!ssl_cert_file) {
        spdlog::warn(
                "Unknown certs location for current distribution. If you hit download issues, "
                "use the envvar `SSL_CERT_FILE` to specify the location manually.");
    }

#elif defined(__APPLE__)
    // The homebrew built OpenSSL adds a dependency on having homebrew installed since it looks in there for certs.
    // The default conan OpenSSL is also misconfigured to look for certs in the OpenSSL build folder.
    // macOS provides certs at the following location, so use those in all cases.
    ssl_cert_file = "/etc/ssl/cert.pem";
#endif

    // Update the envvar.
    if (ssl_cert_file) {
        spdlog::info("Assuming cert location is {}", ssl_cert_file);
        setenv("SSL_CERT_FILE", ssl_cert_file, 1);
    }
#endif  // _WIN32
}
#endif  // DORADO_MODELS_HAS_HTTPLIB

}  // namespace

#if DORADO_MODELS_HAS_HTTPLIB
auto Downloader::create_client() {
    set_ssl_cert_file();

    auto http = std::make_unique<httplib::Client>(urls::URL_ROOT);
    http->set_follow_location(true);
    http->set_connection_timeout(20);

    const char* proxy_url = getenv("dorado_proxy");
    const char* ps = getenv("dorado_proxy_port");

    int proxy_port = 3128;
    if (ps) {
        proxy_port = atoi(ps);
    }

    if (proxy_url) {
        spdlog::info("using proxy: {}:{}", proxy_url, proxy_port);
        http->set_proxy(proxy_url, proxy_port);
    }

    http->set_socket_options([](socket_t sock) {
#ifdef __APPLE__
        // Disable SIGPIPE signal generation since it takes down the entire process
        // whereas we can more gracefully handle the EPIPE error.
        int enabled = 1;
        setsockopt(sock, SOL_SOCKET, SO_NOSIGPIPE, reinterpret_cast<char*>(&enabled),
                   sizeof(enabled));
#else
        (void)sock;
#endif
    });

    return http;
}
#endif  // DORADO_MODELS_HAS_HTTPLIB

Downloader::Downloader(fs::path directory) : m_directory(std::move(directory)) {
#if DORADO_MODELS_HAS_HTTPLIB
    m_client = create_client();
#endif
}

Downloader::~Downloader() = default;

bool Downloader::download(const models::ModelInfo& model) {
    auto archive = m_directory / (model.name + ".zip");

    // Try and download using the native approach, falling back on httplib then on system curl.
    bool success = false;
#if DORADO_MODELS_HAS_FOUNDATION
    if (!success) {
        success = download_foundation(model, archive);
    }
#endif
#if DORADO_MODELS_HAS_HTTPLIB
    if (!success) {
        success = download_httplib(model, archive);
    }
#endif
#if DORADO_MODELS_HAS_CURL_EXE
    if (!success) {
        success = download_curl(model, archive);
    }
#endif

    // Extract it.
    if (success) {
        extract(archive);
    }
    return success;
}

std::string Downloader::get_url(const std::string& model) const {
    return urls::URL_ROOT + urls::URL_PATH + model + ".zip";
}

bool Downloader::validate_checksum(std::string_view data, const models::ModelInfo& info) const {
    // Check that this matches the hash we expect.
    const auto checksum = calculate_checksum(data);
    if (checksum != info.checksum) {
        spdlog::error("Model download failed checksum validation: {} - {} != {}", info.name,
                      checksum, info.checksum);
        return false;
    }
    return true;
}

void Downloader::extract(const fs::path& archive) const {
    spdlog::trace("Extracting model archive: '{}'.", archive.u8string());

    try {
        elz::extractZip(archive, m_directory);
    } catch (const elz::zip_exception& e) {
        spdlog::error("Failed to unzip model archive: '{}'.", e.what());
        throw;
    }

    fs::remove(archive);
}

#if DORADO_MODELS_HAS_HTTPLIB
bool Downloader::download_httplib(const models::ModelInfo& model, const fs::path& archive) {
    spdlog::info(" - downloading {} with httplib", model.name);
    httplib::Result res = m_client->Get(get_url(model.name));
    if (!res) {
        spdlog::error("Failed to download {}: {}", model.name, to_string(res.error()));
        return false;
    }

    // Validate it.
    if (!validate_checksum(res->body, model)) {
        return false;
    }

    // Save it.
    std::ofstream output(archive.string(), std::ofstream::binary);
    output << res->body;
    output.close();
    if (!output) {
        spdlog::error("Failed to save downloaded file to disk");
        return false;
    }

    // Check that all of it was saved.
    if (fs::file_size(archive) != res->body.size()) {
        spdlog::error("Size mismatch between file in memory and file on disk");
        return false;
    }

    return true;
}
#endif  // DORADO_MODELS_HAS_HTTPLIB

#if DORADO_MODELS_HAS_CURL_EXE
bool Downloader::download_curl(const models::ModelInfo& model, const fs::path& archive) {
    spdlog::info(" - downloading {} with curl", model.name);

    // Note: it's safe to call system() here since we're only going to be called with known models.
    std::string args = "curl -L " + get_url(model.name) + " -o " + archive.string();
    errno = 0;
    int ret = system(args.c_str());
    if (ret != 0) {
        spdlog::error("Failed to download {}: ret={}, errno={}", model.name, ret, errno);
        return false;
    }

    // Load it back in and checksum it.
    // Note: there's TOCTOU issues here wrt the download above, and the file_size() call.
    std::ifstream output(archive.string(), std::ofstream::binary);
    std::string buffer;
    buffer.resize(fs::file_size(archive));
    output.read(buffer.data(), buffer.size());
    output.close();
    return validate_checksum(buffer, model);
}
#endif  // DORADO_MODELS_HAS_CURL_EXE

}  // namespace dorado::model_downloader
