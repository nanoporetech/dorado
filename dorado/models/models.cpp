#include "models.h"

#include <elzip/elzip.hpp>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

namespace dorado::models {

namespace {

namespace urls {

const std::string URL_ROOT = "https://cdn.oxfordnanoportal.com";
const std::string URL_PATH = "/software/analysis/dorado/";

}  // namespace urls

// Serialised, released models
namespace simplex {

const ModelMap models = {

        // v3.{3,4,6}
        {"dna_r9.4.1_e8_fast@v3.4",
         {"879cbe2149d5eea524e8902a2d00b39c9b999b66ef40938f0cc37e7e0dc88aed"}},
        {"dna_r9.4.1_e8_hac@v3.3",
         {"6f74b6a90c70cdf984fed73798f5e5a8c17c9af3735ef49e83763143c8c67066"}},
        {"dna_r9.4.1_e8_sup@v3.3",
         {"5fc46541ad4d82b37778e87e65ef0a36b578b1d5b0c55832d80b056bee8703a4"}},
        {"dna_r9.4.1_e8_sup@v3.6",
         {"1db1377b516c158b5d2c39533ac62e8e334e70fcb71c0a4d29e7b3e13632aa73"}},

        // v3.5.2
        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2",
         {"d2c9da317ca431da8adb9ecfc48f9b94eca31c18074062c0e2a8e2e19abc5c13"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2",
         {"c3d4e017f4f7200e9622a55ded303c98a965868e209c08bb79cbbef98ffd552f"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2",
         {"51d30879dddfbf43f794ff8aa4b9cdf681d520cc62323842c2b287282326b4c5"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2",
         {"8d753ac1c30100a49928f7a722f18b14309b5d3417b5f12fd85200239058c36f"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2",
         {"42e790cbb436b7298309d1e8eda7367e1de3b9c04c64ae4da8a28936ec5169f8"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2",
         {"4548b2e25655ce205f0e6fd851bc28a67d9dc13fea7d86efc00c26f227fa17ef"}},

        // v4.0.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.0.0",
         {"d79e19db5361590b44abb2b72395cc83fcca9f822eb3ce049c9675d5d87274dd"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.0.0",
         {"b523f6765859f61f48a2b65c061b099893f78206fe2e5d5689e4aebd6bf42adf"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.0.0",
         {"7c3ab8a1dd89eab53ff122d7e76ff31acdb23a2be988eec9384c6a6715252e41"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0",
         {"d826ccb67c483bdf27ad716c35667eb4335d9487a69e1ac87437c6aabd1f849e"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0",
         {"b04a14de1645b1a0cf4273039309d19b66f7bea9d24bec1b71a58ca20c19d7a0"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0",
         {"a6ca3afac78a25f0ec876f6ea507f42983c7da601d14314515c271551aef9b62"}},

        // v4.1.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.1.0",
         {"5194c533fbdfbab9db590997e755501c65b609c5933943d3099844b83def95b5"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.1.0",
         {"0ba074e95a92e2c4912dbe2c227c5fa5a51e6900437623372b50d4e58f04b9fb"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.1.0",
         {"c236b2a1c0a1c7e670f7bd07e6fd570f01a366538f7f038a76e9cafa62bbf7a4"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.1.0",
         {"8a3d79e0163003591f01e273877cf936a344c8edc04439ee5bd65e0419d802f2"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.1.0",
         {"7da27dc97d45063f0911eac3f08c8171b810b287fd698a4e0c6b1734f02521bf"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.1.0",
         {"47d8d7712341affd88253b5b018609d0caeb76fd929a8dbd94b35c1a2139e37d"}},

        // v4.2.0
        {"dna_r10.4.1_e8.2_400bps_fast@v4.2.0",
         {"be62b912cdabb77b4a25ac9a83ee64ddd8b7fc75deaeb6975f5809c4a97d9c4b"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0",
         {"859d12312cbf47a0c7a8461c26b507e6764590c477e1ea0605510022bbaa8347"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
         {"87c8d044698e37dae1f9100dc4ed0567c6754dcffae446b5ac54a02c0efc401a"}},

        // RNA002
        {"rna002_70bps_fast@v3",
         {"f8f533797e9bf8bbb03085568dc0b77c11932958aa2333902cf2752034707ee6"}},
        {"rna002_70bps_hac@v3",
         {"342b637efdf1a106107a1f2323613f3e4793b5003513b0ed85f6c76574800b52"}},

        // RNA004
        {"rna004_130bps_fast@v3.0.1",
         {"2afa5de03f28162dd85b7be4a2dda108be7cc0a19062db7cb8460628aac462c0"}},
        {"rna004_130bps_hac@v3.0.1",
         {"0b57da141fe97a85d2cf7028c0d0b83c24be35451fd2f8bfb6070f82a1443ea0"}},
        {"rna004_130bps_sup@v3.0.1",
         {"dfe3749c3fbede7203db36ab51689c911d623700e6a24198d398ab927dd756a3"}},
};

}  // namespace simplex

namespace stereo {

const ModelMap models = {
        {"dna_r10.4.1_e8.2_4khz_stereo@v1.1",
         {"d434525cbe1fd00adbd7f8a5f0e7f0bf09b77a9e67cd90f037c5ab52013e7974"}},
        {"dna_r10.4.1_e8.2_5khz_stereo@v1.1",
         {"6c16e3917a12ec297a6f5d1dc83c205fc0ac74282fffaf76b765995033e5f3d4"}},
};

}  // namespace stereo

namespace modified {

const std::vector<std::string> mods = {
        "5mC_5hmC", "5mCG", "5mCG_5hmCG", "5mC", "6mA",
};

const ModelMap models = {

        // v3.{3,4}
        {"dna_r9.4.1_e8_fast@v3.4_5mCG@v0.1",
         {"dab18ae409c754ed164c0214b51d61a3b5126f3e5d043cee60da733db3e78b13"}},
        {"dna_r9.4.1_e8_hac@v3.3_5mCG@v0.1",
         {"349f6623dd43ac8a8ffe9b8e1a02dfae215ea0c1daf32120612dbaabb4f3f16d"}},
        {"dna_r9.4.1_e8_sup@v3.3_5mCG@v0.1",
         {"7ee1893b2de195d387184757504aa5afd76d3feda1078dbc4098efe53acb348a"}},

        {"dna_r9.4.1_e8_fast@v3.4_5mCG_5hmCG@v0",
         {"d45f514c82f25e063ae9e9642d62cec24969b64e1b7b9dffb851b09be6e8f01b"}},
        {"dna_r9.4.1_e8_hac@v3.3_5mCG_5hmCG@v0",
         {"4877da66a0ff6935033557a49f6dbc4676e9d7dba767927fec24b2deae3b681f"}},
        {"dna_r9.4.1_e8_sup@v3.3_5mCG_5hmCG@v0",
         {"7ef57e63f0977977033e3e7c090afca237e26fe3c94b950678346a1982f6116a"}},

        // v3.5.2
        {"dna_r10.4.1_e8.2_260bps_fast@v3.5.2_5mCG@v2",
         {"aa019589113e213f8a67c566874c60024584283de3d8a89ba0d0682c9ce8c2fe"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v3.5.2_5mCG@v2",
         {"bdbc238fbd9640454918d2429f909d9404e5897cc07b948a69462a4eec1838e0"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v3.5.2_5mCG@v2",
         {"0b528c5444c2ca4da7e265b846b24a13c784a34b64a7912fb50c14726abf9ae1"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v3.5.2_5mCG@v2",
         {"ac937da0224c481b6dbb0d1691ed117170ed9e7ff619aa7440123b88274871e8"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v3.5.2_5mCG@v2",
         {"50feb8da3f9b22c2f48d1c3e4aa495630b5f586c1516a74b6670092389bff56e"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v3.5.2_5mCG@v2",
         {"614604cb283598ba29242af68a74c5c882306922c4142c79ac2b3b5ebf3c2154"}},

        // v4.0.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.0.0_5mCG_5hmCG@v2",
         {"b4178526838ed148c81c5189c013096768b58e9741c291fce71647613d93063a"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.0.0_5mCG_5hmCG@v2",
         {"9447249b92febf5d856c247d39f2ce0655f9e2d3079c60b926ef1862e285951b"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.0.0_5mCG_5hmCG@v2",
         {"f41b7a8f53332bebedfd28fceba917e45c9a97aa2dbd21017999e3113cfb0dd3"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.0.0_5mCG_5hmCG@v2",
         {"91e242b5f58f2af843d8b7a975a31bcf8ff0a825bb0583783543c218811d427d"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.0.0_5mCG_5hmCG@v2",
         {"6926ae442b86f8484a95905f1c996c3672a76d499d00fcd0c0fbd6bd1f63fbb3"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.0.0_5mCG_5hmCG@v2",
         {"a7700b0e42779bff88ac02d6b5646b82dcfc65a418d83a8f6d8cca6e22e6cf97"}},

        // v4.1.0
        {"dna_r10.4.1_e8.2_260bps_fast@v4.1.0_5mCG_5hmCG@v2",
         {"93c218d04c958f3559e18132977977ce4e8968e072bb003cab2fe05157c4ded0"}},
        {"dna_r10.4.1_e8.2_260bps_hac@v4.1.0_5mCG_5hmCG@v2",
         {"3178eb66d9e3480dae6e2b6929f8077d4e932820e7825c39b12bd8f381b9814a"}},
        {"dna_r10.4.1_e8.2_260bps_sup@v4.1.0_5mCG_5hmCG@v2",
         {"d7a584f3c2abb6065014326201265ccce5657aec38eeca26d6d522a85b1e31cd"}},

        {"dna_r10.4.1_e8.2_400bps_fast@v4.1.0_5mCG_5hmCG@v2",
         {"aa7af48a90752c15a4b5df5897035629b2657ea0fcc2c785de595c24c7f9e93f"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.1.0_5mCG_5hmCG@v2",
         {"4c91b09d047d36dcb22e43b2fd85ef79e77b07009740ca5130a6a111aa60cacc"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.1.0_5mCG_5hmCG@v2",
         {"73d20629445d21a27dc18a2622063a5916cb04938aa6f12c97ae6b77a883a832"}},

        // v4.2.0
        {"dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2",
         {"a01761e709fd6c114b09ffc7100efb52c37faa38a3f8b281edf405904f04fefa"}},
        {"dna_r10.4.1_e8.2_400bps_hac@v4.2.0_5mCG_5hmCG@v2",
         {"2112aa355757906bfb815bf178fee260ad90cd353781ee45c121024c5caa7c6b"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v2",
         {"6b3604799d85e81d06c97181af093b30483cec9ad02f54a631eca5806f7848ef"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mCG_5hmCG@v3",
         {"9aad5395452ed49fb8442892a8b077afacb80664cf21cc442de76e820ed6e09c"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2",
         {"61ecdba6292637942bc9f143180054084f268d4f8a7e1c7a454413519d5458a7"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2",
         {"0f268e2af4db1023217ee01f2e2e23d47865fde5a5944d915fdb7572d92c0cb5"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v3",
         {"903fb89e7c8929a3a66abf60eb6f1e1a7ab7b7e4a0c40f646dc0b13d5588174c"}},
        {"dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC_5hmC@v1",
         {"28d82762af14e18dd36fb1d9f044b1df96fead8183d3d1ef47a5e92048a2be27"}}

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

std::string calculate_checksum(std::string_view data) {
    // Hash the data.
    std::array<unsigned char, SHA256_DIGEST_LENGTH> hash{};
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash.data());

    // Stringify it.
    std::ostringstream checksum;
    checksum << std::hex;
    checksum.fill('0');
    for (unsigned char byte : hash) {
        checksum << std::setw(2) << static_cast<int>(byte);
    }
    return std::move(checksum).str();
}

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

class ModelDownloader {
    httplib::Client m_client;
    const fs::path m_directory;

    static httplib::Client create_client() {
        set_ssl_cert_file();

        httplib::Client http(urls::URL_ROOT);
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

        return http;
    }

    std::string get_url(const std::string& model) const {
        return urls::URL_ROOT + urls::URL_PATH + model + ".zip";
    }

    void extract(const fs::path& archive) {
        elz::extractZip(archive, m_directory);
        fs::remove(archive);
    }

    bool download_httplib(const std::string& model,
                          const ModelInfo& info,
                          const fs::path& archive) {
        spdlog::info(" - downloading {} with httplib", model);
        httplib::Result res = m_client.Get(get_url(model));
        if (!res) {
            spdlog::error("Failed to download {}: {}", model, to_string(res.error()));
            return false;
        }

        // Check that this matches the hash we expect.
        const auto checksum = calculate_checksum(res->body);
        if (checksum != info.checksum) {
            spdlog::error("Model download failed checksum validation: {} - {} != {}", model,
                          checksum, info.checksum);
            return false;
        }

        // Save it.
        std::ofstream output(archive.string(), std::ofstream::binary);
        output << res->body;
        output.close();
        return true;
    }

    bool download_curl(const std::string& model, const ModelInfo& info, const fs::path& archive) {
        spdlog::info(" - downloading {} with curl", model);

        // Note: it's safe to call system() here since we're only going to be called with known models.
        std::string args = "curl -L " + get_url(model) + " -o " + archive.string();
        errno = 0;
        int ret = system(args.c_str());
        if (ret != 0) {
            spdlog::error("Failed to download {}: ret={}, errno={}", model, ret, errno);
            return false;
        }

        // Load it back in and checksum it.
        // Note: there's TOCTOU issues here wrt the download above, and the file_size() call.
        std::ifstream output(archive.string(), std::ofstream::binary);
        std::string buffer;
        buffer.resize(fs::file_size(archive));
        output.read(buffer.data(), buffer.size());
        output.close();

        const auto checksum = calculate_checksum(buffer);
        if (checksum != info.checksum) {
            spdlog::error("Model download failed checksum validation: {} - {} != {}", model,
                          checksum, info.checksum);
            return false;
        }
        return true;
    }

public:
    ModelDownloader(fs::path directory)
            : m_client(create_client()), m_directory(std::move(directory)) {}

    bool download(const std::string& model, const ModelInfo& info) {
        auto archive = m_directory / (model + ".zip");

        // Try and download using httplib, falling back on curl.
        if (!download_httplib(model, info, archive) && !download_curl(model, info, archive)) {
            return false;
        }

        // Extract it.
        extract(archive);
        return true;
    }
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

bool download_models(const std::string& target_directory, const std::string& selected_model) {
    if (!is_valid_model(selected_model)) {
        spdlog::error("Selected model doesn't exist: {}", selected_model);
        return false;
    }

    ModelDownloader downloader(target_directory);

    bool success = true;
    auto download_model_set = [&](const ModelMap& models) {
        for (const auto& [model, info] : models) {
            if (selected_model == "all" || selected_model == model) {
                if (!downloader.download(std::string(model), info)) {
                    success = false;
                }
            }
        }
    };

    download_model_set(simplex::models);
    download_model_set(stereo::models);
    download_model_set(modified::models);

    return success;
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
        if (!download_models(model_dir.u8string(), modification_model)) {
            throw std::runtime_error("Failed to download model: " + modification_model);
        }
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

}  // namespace dorado::models
