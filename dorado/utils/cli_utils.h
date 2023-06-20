// Add some utilities for CLI.
#pragma once

#include "Version.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <stdio.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dorado {

namespace utils {

// Determine the thread allocation for writer and aligner threads
// in dorado aligner.
inline std::pair<int, int> aligner_writer_thread_allocation(int available_threads,
                                                            float writer_thread_fraction) {
    // clamping because we need at least 1 thread for alignment and for writing.
    int writer_threads =
            std::clamp(static_cast<int>(std::floor(writer_thread_fraction * available_threads)), 1,
                       available_threads - 1);
    int aligner_threads = std::clamp(available_threads - writer_threads, 1, available_threads - 1);
    return std::make_pair(aligner_threads, writer_threads);
}

inline bool is_fd_tty(FILE* fd) {
#ifdef _WIN32
    return _isatty(_fileno(fd));
#else
    return isatty(fileno(fd));
#endif
}

inline bool is_fd_pipe(FILE* fd) {
#ifdef _WIN32
    return false;
#else
    struct stat buffer;
    fstat(fileno(fd), &buffer);
    return S_ISFIFO(buffer.st_mode);
#endif
}

inline void add_pg_hdr(sam_hdr_t* hdr, const std::vector<std::string>& args) {
    sam_hdr_add_lines(hdr, "@HD\tVN:1.6\tSO:unknown", 0);

    std::stringstream pg;
    pg << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
    for (const auto& arg : args) {
        pg << " " << arg;
    }
    pg << std::endl;
    sam_hdr_add_lines(hdr, pg.str().c_str(), 0);
}

inline argparse::ArgumentParser parse_internal_options(
        const std::vector<std::string>& unused_args) {
    auto args = unused_args;
    const std::string prog_name = std::string("internal_args");
    argparse::ArgumentParser private_parser(prog_name);
    private_parser.add_argument("--skip-model-compatibility-check")
            .help("(WARNING: For expert users only) Skip model and data compatibility checks.")
            .default_value(false)
            .implicit_value(true);
    private_parser.add_argument("--dump_stats_file")
            .help("Internal processing stats. output filename.")
            .default_value(std::string(""));
    private_parser.add_argument("--dump_stats_filter")
            .help("Internal processing stats. name filter regex.")
            .default_value(std::string(""));
    args.insert(args.begin(), prog_name);
    private_parser.parse_args(args);

    return private_parser;
}

inline std::tuple<int, int, int> parse_version_str(const std::string& version) {
    size_t first_pos = 0, pos = 0;
    std::vector<int> tokens;
    while ((pos = version.find(".", first_pos)) != std::string::npos) {
        tokens.emplace_back(std::stoi(version.substr(first_pos, pos)));
        first_pos = pos + 1;
    }
    tokens.emplace_back(std::stoi(version.substr(first_pos)));
    if (tokens.size() == 3) {
        return {tokens[0], tokens[1], tokens[2]};
    } else if (tokens.size() == 2) {
        return {tokens[0], tokens[1], 0};
    } else if (tokens.size() == 1) {
        return {tokens[0], 0, 0};
    } else {
        throw std::runtime_error(
                "Could not parse version " + version +
                ". Only version in the format x.y.z where x/y/z are integers is supported");
    }
}

inline uint64_t parse_string_to_size(const std::string& num_str) {
    // check if last character in K, M or G.
    char last_char = num_str[num_str.length() - 1];
    uint64_t multiplier = 1;
    bool is_last_char_alpha = isalpha(last_char);
    if (is_last_char_alpha) {
        switch (last_char) {
        case 'K':
            multiplier = 1e3;
            break;
        case 'M':
            multiplier = 1e6;
            break;
        case 'G':
            multiplier = 1e9;
            break;
        default:
            throw std::runtime_error("Unknown size " + std::to_string(last_char) +
                                     " found. Please choose between K, M or G");
        }
    }
    uint64_t size_num =
            std::stoul(is_last_char_alpha ? num_str.substr(0, num_str.length() - 1) : num_str);
    return size_num * multiplier;
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

}  // namespace utils

}  // namespace dorado
