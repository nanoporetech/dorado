#pragma once

#include "Minimap2Options.h"
#include "utils/arg_parse_ext.h"

#include <string>

namespace dorado::alignment {

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

}  // namespace dorado::alignment