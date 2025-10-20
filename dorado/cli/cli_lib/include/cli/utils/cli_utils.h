// Add some utilities for CLI.
#pragma once

#include "config/BatchParams.h"
#include "dorado_version.h"
#include "hts_utils/bam_utils.h"
#include "torch_utils/auto_detect_device.h"

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif

#include "utils/dev_utils.h"

#include <argparse/argparse.hpp>
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

inline void parse(argparse::ArgumentParser& parser, const std::vector<std::string>& arguments) {
    parser.add_argument("--devopts")
            .hidden()
            .help("Internal options for testing & debugging, 'key=value' pairs separated by ';'")
            .default_value(std::string(""));
    parser.parse_args(arguments);
    utils::details::extract_dev_options(parser.get<std::string>("--devopts"));
}

inline void parse(argparse::ArgumentParser& parser, int argc, const char* const argv[]) {
    return parse(parser, {argv, argv + argc});
}

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
        pg << " " << std::quoted(arg);
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

inline void add_internal_arguments(argparse::ArgumentParser& parser) {
    parser.add_argument("--skip-model-compatibility-check")
            .hidden()
            .help("(WARNING: For expert users only) Skip model and data compatibility checks.")
            .flag();
    parser.add_argument("--dump_stats_file")
            .hidden()
            .help("Internal processing stats. output filename.")
            .default_value(std::string(""));
    parser.add_argument("--dump_stats_filter")
            .hidden()
            .help("Internal processing stats. name filter regex.")
            .default_value(std::string(""));
    parser.add_argument("--run-batchsize-benchmarks")
            .hidden()
            .help("run auto batchsize selection benchmarking instead of using cached benchmark "
                  "figures.")
            .flag();
    parser.add_argument("--emit-batchsize-benchmarks")
            .hidden()
            .help("Write out a CSV and CPP file to the working directory with the auto batchsize "
                  "selection performance stats. Implies --run-batchsize-benchmarks")
            .flag();
    parser.add_argument("--run-for")
            .hidden()
            .help("For performance benchmarking. Run for the specified number of seconds, then "
                  "output performance statistics, and terminate the application.")
            .default_value(0)
            .scan<'i', int>();
}

inline std::vector<std::string> extract_token_from_cli(const std::string& cmd) {
    std::stringstream ss(cmd);
    std::string token;
    std::vector<std::string> tokens;
    while (ss >> std::quoted(token)) {
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
inline std::optional<T> get_optional_argument(const std::string& arg_name,
                                              const argparse::ArgumentParser& parser) {
    static_assert(std::is_default_constructible_v<T>, "T must be default constructible");
    return parser.is_used(arg_name) ? std::optional<T>(parser.get<T>(arg_name)) : std::nullopt;
}

constexpr inline std::string_view DEVICE_HELP{
        "Specify CPU or GPU device: 'auto', 'cpu', 'cuda:all' or "
        "'cuda:<device_id>[,<device_id>...]'. Specifying 'auto' will choose either 'cpu', 'metal' "
        "or 'cuda:all' depending on the presence of a GPU device."};
constexpr inline std::string_view AUTO_DETECT_DEVICE{"auto"};

inline void add_device_arg(argparse::ArgumentParser& parser) {
    parser.add_argument("-x", "--device")
            .help(std::string{DEVICE_HELP})
            .default_value(std::string{AUTO_DETECT_DEVICE});
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

#if DORADO_CUDA_BUILD
inline void log_requested_cuda_devices(const std::vector<utils::CUDADeviceInfo>& cuda_device_info) {
    if (!cuda_device_info.empty()) {
        spdlog::info("Using CUDA devices:");
        for (const auto& device_info : cuda_device_info) {
            spdlog::info("cuda:{} - {}", device_info.device_id, device_info.device_properties.name);
        }
    }
}

inline void log_requested_cuda_devices(const std::string& device_string) {
    auto cuda_device_info = utils::get_cuda_device_info(device_string, false);
    log_requested_cuda_devices(cuda_device_info);
}

#endif

inline config::BatchParams get_batch_params(const argparse::ArgumentParser& parser) {
    config::BatchParams p;
    p.update(config::BatchParams::Priority::CLI_ARG,
             cli::get_optional_argument<int>("--chunksize", parser),
             cli::get_optional_argument<int>("--overlap", parser),
             cli::get_optional_argument<int>("--batchsize", parser));
    return p;
}

inline std::string parse_device(const argparse::ArgumentParser& parser) {
    auto device{parser.get<std::string>("-x")};
    if (!cli::validate_device_string(device)) {
        throw std::runtime_error("Failed to parse --device string.");
    }

    if (device == cli::AUTO_DETECT_DEVICE) {
        return utils::get_auto_detected_device();
    }

    return device;
}

}  // namespace cli

}  // namespace dorado
