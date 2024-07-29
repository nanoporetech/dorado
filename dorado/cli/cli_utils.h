// Add some utilities for CLI.
#pragma once

#include "data_loader/DataLoader.h"
#include "dorado_version.h"
#include "utils/arg_parse_ext.h"
#include "utils/bam_utils.h"

#ifdef _WIN32
// Unreachable code warnings are emitted from argparse, even though they should be disabled by the
// MSVC /external:W0 setting.  This is a limitation of /external: for some C47XX backend warnings.  See:
// https://learn.microsoft.com/en-us/cpp/build/reference/external-external-headers-diagnostics?view=msvc-170#limitations
#pragma warning(push)
#pragma warning(disable : 4702)
#endif  // _WIN32
#include <argparse.hpp>
#ifdef _WIN32
#pragma warning(pop)
#endif  // _WIN32

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dorado {

namespace cli {

// Determine the thread allocation for writer and aligner threads
// in dorado aligner.
inline std::pair<int, int> worker_vs_writer_thread_allocation(int available_threads,
                                                              float writer_thread_fraction) {
    // clamping because we need at least 1 thread for alignment and for writing.
    int writer_threads =
            std::clamp(static_cast<int>(std::floor(writer_thread_fraction * available_threads)), 1,
                       available_threads - 1);
    int aligner_threads = std::clamp(available_threads - writer_threads, 1, available_threads - 1);
    return std::make_pair(aligner_threads, writer_threads);
}

inline void add_pg_hdr(sam_hdr_t* hdr,
                       const std::string& pg_id,
                       const std::vector<std::string>& args,
                       const std::string& device) {
    utils::add_hd_header_line(hdr);
    auto safe_id = sam_hdr_pg_id(hdr, pg_id.c_str());

    std::stringstream pg;
    pg << "@PG\tID:" << safe_id << "\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
    for (const auto& arg : args) {
        pg << " " << arg;
    }

#if DORADO_CUDA_BUILD
    auto gpu_string = utils::get_cuda_gpu_names(device);
    if (!gpu_string.empty()) {
        pg << "\tDS:gpu:" << gpu_string;
    }
#else
    (void)device;
#endif

    pg << '\n';
    sam_hdr_add_lines(hdr, pg.str().c_str(), 0);
}

inline void add_internal_arguments(utils::arg_parse::ArgParser& parser) {
    parser.hidden.add_argument("--skip-model-compatibility-check")
            .help("(WARNING: For expert users only) Skip model and data compatibility checks.")
            .default_value(false)
            .implicit_value(true);
    parser.hidden.add_argument("--dump_stats_file")
            .help("Internal processing stats. output filename.")
            .default_value(std::string(""));
    parser.hidden.add_argument("--dump_stats_filter")
            .help("Internal processing stats. name filter regex.")
            .default_value(std::string(""));
}

inline std::vector<std::string> extract_token_from_cli(const std::string& cmd) {
    std::stringstream ss(cmd);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, ' ')) {
        tokens.push_back(token);
    }
    if (tokens.size() < 4) {
        throw std::runtime_error(
                "Cmdline requires at least 4 tokens including binary name, found: " + cmd);
    }
    return tokens;
}

// ArgumentParser has no method returning optional arguments with default values so this function returns
// optional<T> which lets us determine if the value (including the default value) was explictly set by the user.
template <typename T>
inline std::optional<T> get_optional_argument(const std::string arg_name,
                                              const argparse::ArgumentParser& parser) {
    static_assert(std::is_default_constructible_v<T>, "T must be default constructible");
    return parser.is_used(arg_name) ? std::optional<T>(parser.get<T>(arg_name)) : std::nullopt;
}

constexpr inline std::string_view DEVICE_HELP{
        "Specify CPU or GPU device: 'auto', 'cpu', 'cuda:all' or "
        "'cuda:<device_id>[,<device_id>...]'. Specifying 'auto' will choose either 'cpu', 'metal' "
        "or 'cuda:all' depending on the presence of a GPU device."};
constexpr inline std::string_view AUTO_DETECT_DEVICE{"auto"};

inline void add_device_arg(utils::arg_parse::ArgParser& parser, const std::string& default_device) {
    parser.visible.add_argument("-x", "--device")
            .help(std::string{DEVICE_HELP})
            .default_value(default_device);
}

inline bool validate_device_string(const std::string& device) {
    if (device == "cpu" || device == AUTO_DETECT_DEVICE) {
        return true;
    }
#if DORADO_METAL_BUILD
    if (device == "metal") {
        return true;
    }
#elif DORADO_CUDA_BUILD
    if (!device.empty() && device.substr(0, 5) == "cuda:") {
        std::string error_message{};
        std::vector<std::string> devices{};
        if (utils::try_parse_cuda_device_string(device, devices, error_message)) {
            return true;
        }
        spdlog::error(error_message);
        return false;
    }
#endif
    spdlog::error("Invalid device string: {}\n{}", device, DEVICE_HELP);
    return false;
}

inline std::string get_auto_detected_device() {
#if DORADO_METAL_BUILD
    return "metal";
#elif DORADO_CUDA_BUILD
    return torch::cuda::is_available() ? "cuda:all" : "cpu";
#else
    return "cpu";
#endif
}

}  // namespace cli

}  // namespace dorado
