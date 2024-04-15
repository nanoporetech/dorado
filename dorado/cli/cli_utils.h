// Add some utilities for CLI.
#pragma once

#include "dorado_version.h"
#include "models/kits.h"
#include "utils/dev_utils.h"

#include <optional>
#include <stdexcept>

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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

namespace cli {

static constexpr auto HIDDEN_PROGRAM_NAME = "internal_args";

struct ArgParser {
    ArgParser(std::string program_name)
            : visible(std::move(program_name), DORADO_VERSION, argparse::default_arguments::help),
              hidden(HIDDEN_PROGRAM_NAME){};
    argparse::ArgumentParser visible;
    argparse::ArgumentParser hidden;
};

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
                       const std::vector<std::string>& args,
                       const std::string& device) {
    sam_hdr_add_lines(hdr, "@HD\tVN:1.6\tSO:unknown", 0);

    std::stringstream pg;
    pg << "@PG\tID:basecaller\tPN:dorado\tVN:" << DORADO_VERSION << "\tCL:dorado";
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

inline std::tuple<int, int, int> parse_version_str(const std::string& version) {
    size_t first_pos = 0, pos = 0;
    std::vector<int> tokens;
    while ((pos = version.find('.', first_pos)) != std::string::npos) {
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
std::vector<T> parse_string_to_sizes(const std::string& str) {
    std::vector<T> sizes;
    const char* c_str = str.c_str();
    char* p;
    while (true) {
        double x = strtod(c_str, &p);
        if (p == c_str) {
            throw std::runtime_error("Cannot parse size '" + str + "'.");
        }
        if (*p == 'G' || *p == 'g') {
            x *= 1e9;
            ++p;
        } else if (*p == 'M' || *p == 'm') {
            x *= 1e6;
            ++p;
        } else if (*p == 'K' || *p == 'k') {
            x *= 1e3;
            ++p;
        }
        sizes.emplace_back(static_cast<T>(std::round(x)));
        if (*p == ',') {
            c_str = ++p;
            continue;
        } else if (*p == 0) {
            break;
        }
        throw std::runtime_error("Unknown suffix '" + std::string(p) + "'.");
    }
    return sizes;
}

template <class T = uint64_t>
T parse_string_to_size(const std::string& str) {
    return parse_string_to_sizes<T>(str)[0];
}

inline bool parse_yes_or_no(const std::string& str) {
    if (str == "yes" || str == "y") {
        return true;
    }
    if (str == "no" || str == "n") {
        return false;
    }
    auto msg = "Unsupported value '" + str + "'; option only accepts '(y)es' or '(n)o'.";
    throw std::runtime_error(msg);
}

inline std::string to_size(double value) {
    std::stringstream res;
    if (value < 1e3) {
        res << value;
    } else if (value < 1e6) {
        res << value / 1e3 << 'K';
    } else if (value < 1e9) {
        res << value / 1e6 << 'M';
    } else {
        res << value / 1e9 << 'G';
    }
    return res.str();
}

inline std::string to_yes_or_no(bool value) { return value ? "yes" : "no"; }

inline void add_internal_arguments(ArgParser& parser) {
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

template <class Options>
void add_minimap2_arguments(ArgParser& parser, const Options& dflt) {
    parser.visible.add_argument("-k")
            .help("minimap2 k-mer size for alignment (maximum 28).")
            .template default_value<int>(dflt.kmer_size)
            .template scan<'i', int>();

    parser.visible.add_argument("-w")
            .help("minimap2 minimizer window size for alignment.")
            .template default_value<int>(dflt.window_size)
            .template scan<'i', int>();

    parser.visible.add_argument("-I")
            .help("minimap2 index batch size.")
            .default_value(to_size(double(dflt.index_batch_size)));

    parser.visible.add_argument("--secondary")
            .help("minimap2 outputs secondary alignments")
            .default_value(to_yes_or_no(dflt.print_secondary));

    parser.visible.add_argument("-N")
            .help("minimap2 retains at most INT secondary alignments")
            .default_value(dflt.best_n_secondary)
            .template scan<'i', int>();

    parser.visible.add_argument("-Y")
            .help("minimap2 uses soft clipping for supplementary alignments")
            .default_value(false)
            .implicit_value(true);

    parser.visible.add_argument("--bandwidth")
            .help("minimap2 chaining/alignment bandwidth and optionally long-join bandwidth "
                  "specified as NUM,[NUM]")
            .default_value(to_size(dflt.bandwidth) + "," + to_size(dflt.bandwidth_long));

    // Setting options to lr:hq which is appropriate for high quality nanopore reads.
    parser.visible.add_argument("--mm2-preset")
            .help("minimap2 preset for indexing and mapping. Alias for the -x "
                  "option in minimap2.")
            .default_value(dflt.mm2_preset);

    parser.hidden.add_argument("--secondary-seq")
            .help("minimap2 output seq/qual for secondary and supplementary alignments")
            .default_value(false)
            .implicit_value(true);

    parser.hidden.add_argument("--print-aln-seq")
            .help("minimap2 debug print qname and aln_seq")
            .default_value(false)
            .implicit_value(true);
}

inline void parse(ArgParser& parser, int argc, const char* const argv[]) {
    parser.hidden.add_argument("--devopts")
            .help("Internal options for testing & debugging, 'key=value' pairs separated by ';'")
            .default_value(std::string(""));
    auto remaining_args = parser.visible.parse_known_args(argc, argv);
    remaining_args.insert(remaining_args.begin(), HIDDEN_PROGRAM_NAME);
    parser.hidden.parse_args(remaining_args);
    utils::details::extract_dev_options(parser.hidden.get<std::string>("--devopts"));
}

template <class Options>
Options process_minimap2_arguments(const ArgParser& parser, const Options& dflt) {
    Options res;
    res.kmer_size = short(parser.visible.get<int>("k"));
    res.window_size = short(parser.visible.get<int>("w"));
    res.index_batch_size = cli::parse_string_to_size(parser.visible.get<std::string>("I"));
    res.print_secondary = cli::parse_yes_or_no(parser.visible.get<std::string>("secondary"));
    res.best_n_secondary = parser.visible.get<int>("N");
    if (res.best_n_secondary == 0) {
        spdlog::warn("Changed '-N 0' to '-N {} --secondary=no'", dflt.best_n_secondary);
        res.print_secondary = false;
        res.best_n_secondary = dflt.best_n_secondary;
    }
    auto bandwidth = cli::parse_string_to_sizes(parser.visible.get<std::string>("--bandwidth"));
    switch (bandwidth.size()) {
    case 1:
        res.bandwidth = int(bandwidth[0]);
        res.bandwidth_long = dflt.bandwidth_long;
        break;
    case 2:
        res.bandwidth = int(bandwidth[0]);
        res.bandwidth_long = int(bandwidth[1]);
        break;
    default:
        throw std::runtime_error("Wrong number of arguments for option '-r'.");
    }
    res.soft_clipping = parser.visible.get<bool>("Y");
    res.mm2_preset = parser.visible.get<std::string>("mm2-preset");
    res.secondary_seq = parser.hidden.get<bool>("secondary-seq");
    res.print_aln_seq = parser.hidden.get<bool>("print-aln-seq");
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
