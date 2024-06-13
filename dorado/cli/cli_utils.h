// Add some utilities for CLI.
#pragma once

#include "dorado_version.h"
#include "models/kits.h"
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
#include "data_loader/ModelFinder.h"

#if DORADO_CUDA_BUILD
#include "utils/cuda_utils.h"
#endif

#include <htslib/sam.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
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

inline ModelSelection parse_model_argument(const std::string& model_arg) {
    try {
        return ModelComplexParser::parse(model_arg);
    } catch (std::exception& e) {
        spdlog::error("Failed to parse model argument. {}", e.what());
        std::exit(EXIT_FAILURE);
    }
}

inline ModelFinder model_finder(const ModelSelection& model_selection,
                                const std::filesystem::path& data,
                                bool recursive,
                                bool suggestions) {
    try {
        return ModelFinder(ModelFinder::inspect_chemistry(data.u8string(), recursive),
                           model_selection, suggestions);
    } catch (std::exception& e) {
        spdlog::error(e.what());
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace cli

}  // namespace dorado
