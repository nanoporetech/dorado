#include "minimap2_args.h"

#include "minimap2_wrappers.h"
#include "utils/string_utils.h"

#include <minimap.h>
#include <mmpriv.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace {

void mm_mapopt_override(mm_mapopt_t* mapopt) {
    // Force cigar generation.
    mapopt->flag |= MM_F_CIGAR;

    // Equivalent to "--cap-kalloc 100m --cap-sw-mem 50m"
    mapopt->cap_kalloc = 100'000'000;
    mapopt->max_sw_mat = 50'000'000;
}

void mm_idxopt_override(mm_idxopt_t* idxopt) {
    idxopt->batch_size = 16000000000;
    idxopt->mini_batch_size = idxopt->batch_size;
}

const mm_mapopt_t& mm_mapopt_default() {
    static const mm_mapopt_t instance = [] {
        mm_mapopt_t mapopt;
        mm_mapopt_init(&mapopt);
        mm_mapopt_override(&mapopt);
        return mapopt;
    }();
    return instance;
}

const mm_idxopt_t mm_idxopt_default() {
    static const mm_idxopt_t instance = [] {
        mm_idxopt_t idxopt{};
        mm_idxopt_init(&idxopt);
        mm_idxopt_override(&idxopt);
        return idxopt;
    }();
    return instance;
}

template <typename TO, typename FROM>
std::optional<TO> get_optional_as(const std::optional<FROM>& from_optional) {
    if (from_optional) {
        return std::make_optional(static_cast<TO>(*from_optional));
    } else {
        return std::nullopt;
    }
}

}  // namespace

namespace dorado::alignment::minimap2 {

std::string extract_mm2_opts_arg(const std::vector<std::string>& args,
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

void add_mm2_opts_arg(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(MM2_OPTS_ARG)
            .help("Optional minimap2 options string. For multiple arguments surround with double "
                  "quotes.");
}

void add_arguments(utils::arg_parse::ArgParser& parser) {
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
            .default_value(std::string{DEFAULT_MM_PRESET});

    parser.hidden.add_argument("--secondary-seq")
            .help("minimap2 output seq/qual for secondary and supplementary alignments")
            .default_value(false)
            .implicit_value(true);

    parser.hidden.add_argument("--print-aln-seq")
            .help("minimap2 debug print qname and aln_seq")
            .default_value(false)
            .implicit_value(true);
}

void apply_preset(Minimap2Options& options, const std::string& preset) {
    if (preset.empty() || mm_set_opt(preset.c_str(), &options.index_options->get(),
                                     &options.mapping_options->get()) == 0) {
        return;
    }
    throw std::runtime_error("Cannot set mm2 options with preset: " + preset);
}

void apply_indexing_options(const utils::arg_parse::ArgParser& parser, mm_idxopt_t& options) {
    auto kmer = get_optional_as<short>(parser.visible.present<int>("k"));
    if (kmer) {
        options.k = *kmer;
    }
    auto window_size = get_optional_as<short>(parser.visible.present<int>("w"));
    if (window_size) {
        options.w = *window_size;
    }
    auto index_batch_size = parser.visible.present<std::string>("I");
    if (index_batch_size) {
        options.batch_size = utils::arg_parse::parse_string_to_size<uint64_t>(*index_batch_size);
    }
    options.mini_batch_size = options.batch_size;
}

void apply_mapping_options(const utils::arg_parse::ArgParser& parser, mm_mapopt_t& options) {
    auto secondary = parser.visible.present<std::string>("--secondary");
    bool print_secondary{true};
    if (secondary && !utils::arg_parse::parse_yes_or_no(*secondary)) {
        print_secondary = false;
    }

    auto best_n_secondary = parser.visible.present<int>("N");
    if (best_n_secondary.value_or(1) == 0) {
        spdlog::warn("Ignoring '-N 0', using preset default");
        print_secondary = true;
    } else {
        options.best_n = best_n_secondary.value_or(options.best_n);
    }

    if (!print_secondary) {
        options.flag |= MM_F_NO_PRINT_2ND;
    }

    auto optional_bandwidth = parser.visible.present<std::string>("--bandwidth");
    if (optional_bandwidth) {
        auto bandwidth = utils::arg_parse::parse_string_to_sizes<int>(*optional_bandwidth);
        switch (bandwidth.size()) {
        case 2:
            options.bw_long = bandwidth[1];
            [[fallthrough]];
        case 1:
            options.bw = bandwidth[0];
            break;
        default:
            throw std::runtime_error(
                    "Wrong number of arguments for minimap2 option '--bandwidth'.");
        }
    }
    auto soft_clipping = parser.visible.present<bool>("Y");
    if (soft_clipping.value_or(false)) {
        options.flag |= MM_F_SOFTCLIP;
    }
    if (parser.hidden.get<bool>("secondary-seq")) {
        options.flag |= MM_F_SECONDARY_SEQ;
    }
}

Minimap2Options process_arguments(const utils::arg_parse::ArgParser& parser) {
    Minimap2Options res{};
    res.index_options->get() = mm_idxopt_default();
    res.mapping_options->get() = mm_mapopt_default();

    // apply any preset first.
    auto preset = parser.visible.get<std::string>("mm2-preset");
    apply_preset(res, preset);

    apply_indexing_options(parser, res.index_options->get());
    apply_mapping_options(parser, res.mapping_options->get());

    auto rc = mm_check_opt(&res.index_options->get(), &res.mapping_options->get());
    if (rc < 0) {
        throw std::runtime_error(
                "Invalid minimap2 options string, for details run with --verbose flag. err_code: " +
                std::to_string(rc));
    }

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
    res.secondary_seq = parser.hidden.get<bool>("secondary-seq");

    auto junc_bed = parser.visible.present<std::string>("--junc-bed");
    if (junc_bed) {
        res.junc_bed = std::move(*junc_bed);
    }
    res.mm2_preset = parser.visible.get<std::string>("mm2-preset");
    res.print_aln_seq = parser.hidden.get<bool>("print-aln-seq");

    //res.bandwidth = mm_mapopt_default().bw;
    //res.kmer_size = mm_idxopt_default().k;

    return res;
}

Minimap2Options process_option_string(const std::string& minimap2_option_string) {
    std::vector<std::string> mm2_args = [&minimap2_option_string] {
        if (minimap2_option_string.empty()) {
            return std::vector<std::string>{"minimap2_options"};
        }
        return utils::split("minimap2_options " + minimap2_option_string, ' ');
    }();

    utils::arg_parse::ArgParser parser("minimap2_options");
    add_arguments(parser);

    try {
        utils::arg_parse::parse(parser, mm2_args);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        throw;
    }

    return process_arguments(parser);
}

void apply_cs_option(Minimap2Options& options, const std::string& cs_opt) {
    if (cs_opt.empty()) {
        return;
    }
    auto& flag = options.mapping_options->get().flag;

    flag |= MM_F_OUT_CS | MM_F_CIGAR;
    if (cs_opt == "short") {
        flag &= ~MM_F_OUT_CS_LONG;
    } else if (cs_opt == "long") {
        flag |= MM_F_OUT_CS_LONG;
    } else if (cs_opt == "none") {
        flag &= ~MM_F_OUT_CS;
    } else {
        spdlog::warn("Unrecognized options for --cs={}", cs_opt);
    }
}

void apply_dual_option(Minimap2Options& options, const std::string& dual) {
    if (dual.empty()) {
        return;
    }

    if (dual == "yes") {
        options.mapping_options->get().flag &= ~MM_F_NO_DUAL;
    } else if (dual == "no") {
        options.mapping_options->get().flag |= MM_F_NO_DUAL;
    } else {
        spdlog::warn("Unrecognized options for --dual={}", dual);
    }
}

}  // namespace dorado::alignment::minimap2