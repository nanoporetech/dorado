#pragma once

#include "Minimap2Options.h"
#include "utils/arg_parse_ext.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace std::literals;

namespace dorado::alignment {

constexpr inline std::string_view MM2_OPTS_ARG = "--mm2-opts";

// call with extract_minimap2_args({argv, argv + argc}, remaining_args);
inline std::string extract_minimap2_args(const std::vector<std::string>& args,
                                         std::vector<std::string>& remaining_args) {
    auto mm2_opt_key_itr = std::find(std::cbegin(args), std::cend(args), MM2_OPTS_ARG);
    if (mm2_opt_key_itr == std::cend(args)) {
        remaining_args = args;
        return {};
    }
    auto mm2_opt_value_itr = mm2_opt_key_itr + 1;
    if (mm2_opt_value_itr == std::cend(args)) {
        throw std::runtime_error("Missing value for " + std::string{MM2_OPTS_ARG} +
                                 " command line argument.");
    }
    remaining_args.insert(std::end(remaining_args), std::cbegin(args), mm2_opt_key_itr);
    remaining_args.insert(std::end(remaining_args), mm2_opt_value_itr + 1, std::cend(args));
    return *mm2_opt_value_itr;
}

inline void add_minimap2_opts_arg(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(MM2_OPTS_ARG)
            .help("Optional minimap2 options string. For multiple arguments surround with double "
                  "quotes.");
}

inline void add_minimap2_arguments(utils::arg_parse::ArgParser& parser,
                                   const std::string& default_preset) {
    parser.visible.add_argument("-k")
            .help("minimap2 k-mer size for alignment (maximum 28).")
            .template scan<'i', int>();

    parser.visible.add_argument("-w")
            .help("minimap2 minimizer window size for alignment.")
            .template scan<'i', int>();

    parser.visible.add_argument("-I").help("minimap2 index batch size.");

    parser.visible.add_argument("--secondary").help("minimap2 outputs secondary alignments");

    parser.visible.add_argument("-N")
            .help("minimap2 retains at most INT secondary alignments")
            .template scan<'i', int>();

    parser.visible.add_argument("-Y")
            .help("minimap2 uses soft clipping for supplementary alignments")
            .implicit_value(true);

    parser.visible.add_argument("--bandwidth")
            .help("minimap2 chaining/alignment bandwidth and optionally long-join bandwidth "
                  "specified as NUM,[NUM]");

    parser.visible.add_argument("--junc-bed")
            .help("Optional file with gene annotations in the BED12 format (aka 12-column BED), or "
                  "intron positions in 5-column BED. With this option, minimap2 prefers splicing "
                  "in annotations.");

    // Setting options to lr:hq which is appropriate for high quality nanopore reads.
    parser.visible.add_argument("--mm2-preset")
            .help("minimap2 preset for indexing and mapping. Alias for the -x "
                  "option in minimap2.")
            .default_value(default_preset);

    parser.hidden.add_argument("--secondary-seq")
            .help("minimap2 output seq/qual for secondary and supplementary alignments")
            .default_value(false)
            .implicit_value(true);

    parser.hidden.add_argument("--print-aln-seq")
            .help("minimap2 debug print qname and aln_seq")
            .default_value(false)
            .implicit_value(true);
}

template <typename TO, typename FROM>
std::optional<TO> get_optional_as(const std::optional<FROM>& from_optional) {
    if (from_optional) {
        return std::make_optional(static_cast<TO>(*from_optional));
    } else {
        return std::nullopt;
    }
}

inline Minimap2Options process_minimap2_arguments(const utils::arg_parse::ArgParser& parser) {
    Minimap2Options res{};
    res.kmer_size = get_optional_as<short>(parser.visible.present<int>("k"));
    res.window_size = get_optional_as<short>(parser.visible.present<int>("w"));
    auto index_batch_size = parser.visible.present<std::string>("I");
    if (index_batch_size) {
        res.index_batch_size = std::make_optional(
                utils::arg_parse::parse_string_to_size<uint64_t>(*index_batch_size));
    }
    auto print_secondary = parser.visible.present<std::string>("--secondary");
    if (print_secondary) {
        res.print_secondary =
                std::make_optional(utils::arg_parse::parse_yes_or_no(*print_secondary));
    }
    res.best_n_secondary = parser.visible.present<int>("N");
    if (res.best_n_secondary.value_or(1) == 0) {
        spdlog::warn("Ignoring '-N 0', using preset default");
        res.print_secondary = std::nullopt;
        res.best_n_secondary = std::nullopt;
    }

    auto optional_bandwidth = parser.visible.present<std::string>("--bandwidth");
    if (optional_bandwidth) {
        auto bandwidth = utils::arg_parse::parse_string_to_sizes<int>(*optional_bandwidth);
        switch (bandwidth.size()) {
        case 1:
            res.bandwidth = std::make_optional<int>(bandwidth[0]);
            break;
        case 2:
            res.bandwidth = std::make_optional<int>(bandwidth[0]);
            res.bandwidth_long = std::make_optional<int>(bandwidth[1]);
            break;
        default:
            throw std::runtime_error("Wrong number of arguments for option '-r'.");
        }
    }
    res.soft_clipping = parser.visible.present<bool>("Y");
    auto junc_bed = parser.visible.present<std::string>("--junc-bed");
    if (junc_bed) {
        res.junc_bed = std::move(*junc_bed);
    }
    res.mm2_preset = parser.visible.get<std::string>("mm2-preset");
    res.secondary_seq = parser.hidden.get<bool>("secondary-seq");
    res.print_aln_seq = parser.hidden.get<bool>("print-aln-seq");
    return res;
}

inline Minimap2Options process_minimap2_option_string(const std::string& minimap2_option_string) {
    auto mm2_args = utils::split("minimap2_options " + minimap2_option_string, ' ');
    utils::arg_parse::ArgParser parser("minimap2_options");
    add_minimap2_arguments(parser, DEFAULT_MM_PRESET);

    try {
        utils::arg_parse::parse(parser, mm2_args);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        throw;
    }

    return process_minimap2_arguments(parser);
}

}  // namespace dorado::alignment