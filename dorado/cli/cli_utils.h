// Add some utilities for CLI.
#pragma once

#include "Version.h"
#include "utils/dev_utils.h"

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
    private_parser.add_argument("--devopts")
            .help("Internal options for testing & debugging, 'key=value' pairs separated by ';'")
            .default_value(std::string(""));
    args.insert(args.begin(), prog_name);
    private_parser.parse_args(args);
    utils::details::extract_dev_options(private_parser.get<std::string>("--devopts"));

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

template <class T = int64_t>
inline std::vector<T> parse_string_to_sizes(const std::string& str,
                                            std::optional<std::string> opt = std::nullopt) {
    std::size_t pos;
    std::vector<T> sizes;
    const char* c_str = str.c_str();
    char* p;
    while (true) {
        double x = strtod(c_str, &p);
        if (p == c_str) {
            auto msg = "Cannot parse size '" + str + "'.";
            if (opt)
                msg = "Error parsing option " + *opt + ": " + msg;
            throw std::runtime_error(msg);
        }
        if (*p == 'G' || *p == 'g') {
            x *= 1e9, ++p;
        } else if (*p == 'M' || *p == 'm') {
            x *= 1e6, ++p;
        } else if (*p == 'K' || *p == 'k') {
            x *= 1e3, ++p;
        }
        sizes.emplace_back(x + .499);
        if (*p == ',') {
            c_str = ++p;
            continue;
        } else if (*p == 0) {
            break;
        }
        auto msg = "Unknown suffix '" + std::string(p) + "'.";
        if (opt)
            msg = "Error parsing option " + *opt + ": " + msg;
        throw std::runtime_error(msg);
    }
    while (*p != 0)
        ;
    std::cout << std::endl;
    return sizes;
}

template <class T = int>
inline T parse_string_to_size(const std::string& str,
                              std::optional<std::string> opt = std::nullopt) {
    return parse_string_to_sizes<T>(str, opt)[0];
}

inline bool parse_yes_or_no(const std::string& str, std::optional<std::string> opt = std::nullopt) {
    if (str == "yes" || str == "y")
        return true;
    if (str == "no" || str == "n")
        return false;
    auto msg = "Unsupported value '" + str + "'; option  only accepts 'yes' or 'no'.";
    if (opt)
        msg = "Error parsing option " + *opt + ": " + msg;
    throw std::runtime_error(msg);
}

inline std::string to_size(double value) {
    if (value < 1e3)
        return std::to_string(value);
    if (value < 1e6)
        return std::to_string(value / 1e3) + "K";
    if (value < 1e9)
        return std::to_string(value / 1e6) + "M";
    return std::to_string(value / 1e9) + "G";
}

inline std::string to_yes_or_no(bool value) { return value ? "yes" : "no"; }

template <class Options>
inline void add_minimap2_arguments(argparse::ArgumentParser& parser, const Options& dflt) {
    parser.add_argument("-k")
            .help("minimap2 k-mer size for alignment (maximum 28).")
            .template default_value<int>(dflt.kmer_size)
            .template scan<'i', int>();

    parser.add_argument("-w")
            .help("minimap2 minimizer window size for alignment.")
            .template default_value<int>(dflt.window_size)
            .template scan<'i', int>();

    parser.add_argument("-I")
            .help("minimap2 index batch size.")
            .default_value(to_size(dflt.index_batch_size));

    parser.add_argument("-K", "--mb-size")
            .help("minimap2 minibatch size for mapping")
            .default_value(to_size(dflt.mini_batch_size));

    parser.add_argument("--secondary")
            .help("minimap2 outputs secondary alignments")
            .default_value(to_yes_or_no(dflt.print_secondary));

    parser.add_argument("-N")
            .help("minimap2 retains at most INT secondary alignments")
            .default_value(dflt.best_n_secondary)
            .template scan<'i', int>();

    parser.add_argument("-r")
            .help("minimap2 chaining/alignment bandwidth and long-join bandwidth")
            .default_value(to_size(dflt.bandwidth) + "," + to_size(dflt.bandwidth_long));
}

template <class Options>
inline Options parse_minimap2_arguments(const argparse::ArgumentParser& parser,
                                        const Options& dflt) {
    Options res;
    res.kmer_size = parser.template get<int>("k");
    res.window_size = parser.template get<int>("w");
    res.index_batch_size = cli::parse_string_to_size(parser.template get<std::string>("I"));
    res.mini_batch_size = cli::parse_string_to_size(parser.template get<std::string>("K"));
    res.print_secondary = cli::parse_yes_or_no(parser.template get<std::string>("secondary"));
    res.best_n_secondary = parser.template get<int>("N");
    if (res.best_n_secondary == 0) {
        spdlog::warn("Changed '-N 0' to '-N {} --secondary=no'", dflt.best_n_secondary);
        res.print_secondary = false;
        res.best_n_secondary = dflt.best_n_secondary;
    }
    auto bandwidth = cli::parse_string_to_sizes(parser.template get<std::string>("r"));
    switch (bandwidth.size()) {
    case 1:
        res.bandwidth = bandwidth[0];
        res.bandwidth_long = dflt.bandwidth_long;
        break;
    case 2:
        res.bandwidth = bandwidth[0];
        res.bandwidth_long = bandwidth[1];
        break;
    default:
        throw std::runtime_error("Wrong number of arguments for option '-r'.");
    }
    return res;
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

}  // namespace cli

}  // namespace dorado
