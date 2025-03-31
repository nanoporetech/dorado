#include "minimap2_args.h"

#include "minimap2_wrappers.h"
#include "utils/string_utils.h"

#include <minimap.h>
#include <mmpriv.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace {

constexpr inline std::string_view MM2_OPTS_ARG = "--mm2-opts";

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

namespace dorado::alignment::mm2 {

std::string extract_options_string_arg(const std::vector<std::string>& args,
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

void add_options_string_arg(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(MM2_OPTS_ARG)
            .help("Optional minimap2 options string. For multiple arguments surround with double "
                  "quotes.");
}

namespace {

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

    parser.visible.add_argument("-r").help(
            "minimap2 chaining/alignment bandwidth and optionally long-join bandwidth "
            "specified as NUM,[NUM]");

    parser.visible.add_argument("--junc-bed")
            .help("Optional file with gene annotations in the BED12 format (aka 12-column BED), or "
                  "intron positions in 5-column BED. With this option, minimap2 prefers splicing "
                  "in annotations.");

    // Setting options to lr:hq which is appropriate for high quality nanopore reads.
    parser.visible.add_argument("-x")
            .help("minimap2 preset for indexing and mapping.")
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

    auto optional_bandwidth = parser.visible.present<std::string>("-r");
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
                    "Wrong number of arguments for minimap2 bandwidth option '-r'.");
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

std::optional<Minimap2Options> process_arguments(const utils::arg_parse::ArgParser& parser,
                                                 std::string& error_message) {
    Minimap2Options res{};
    res.index_options->get() = mm_idxopt_default();
    res.mapping_options->get() = mm_mapopt_default();

    // apply preset before overwriting with other user supplied options.
    apply_preset(res, parser.visible.get<std::string>("-x"));

    apply_indexing_options(parser, res.index_options->get());
    apply_mapping_options(parser, res.mapping_options->get());

    auto rc = mm_check_opt(&res.index_options->get(), &res.mapping_options->get());
    if (rc < 0) {
        error_message = "Invalid minimap2 options string. Error code: " + std::to_string(rc);
        return std::nullopt;
    }

    // Cache the --junc-bed arg with the index options for use when the index is loaded
    auto junc_bed = parser.visible.present<std::string>("--junc-bed");
    if (junc_bed) {
        res.junc_bed = std::move(*junc_bed);
    }

    if (parser.hidden.get<bool>("print-aln-seq")) {
        // set the global flags
        mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ;
    }

    return res;
}

}  // namespace

std::string get_help_message() {
    utils::arg_parse::ArgParser parser("minimap2_options");
    add_arguments(parser);
    std::ostringstream parser_stream;
    parser_stream << parser.visible;
    return parser_stream.str();
}

Minimap2Options parse_options(const std::string& minimap2_option_string) {
    std::string error_message{};
    auto minimap2_options = try_parse_options(minimap2_option_string, error_message);
    if (!minimap2_options) {
        spdlog::error("{}\n{}", error_message, get_help_message());
        throw std::runtime_error(error_message);
    }
    return *minimap2_options;
}

namespace {

std::optional<Minimap2Options> try_parse_options_impl(utils::arg_parse::ArgParser& parser,
                                                      const std::string& minimap2_option_string,
                                                      std::string& error_message) {
    std::vector<std::string> mm2_args = [&minimap2_option_string] {
        if (minimap2_option_string.empty()) {
            return std::vector<std::string>{"minimap2_options"};
        }
        return utils::split("minimap2_options " + minimap2_option_string, ' ');
    }();
    add_arguments(parser);
    try {
        utils::arg_parse::parse(parser, mm2_args);
    } catch (const std::exception& e) {
        error_message = e.what();
        return std::nullopt;
    }

    return process_arguments(parser, error_message);
}

}  // namespace

std::optional<Minimap2Options> try_parse_options(const std::string& minimap2_option_string,
                                                 std::string& error_message) {
    utils::arg_parse::ArgParser parser("minimap2_options");
    return try_parse_options_impl(parser, minimap2_option_string, error_message);
}

std::optional<Minimap2Options> try_parse_options_no_help(const std::string& minimap2_option_string,
                                                         std::string& error_message) {
    utils::arg_parse::ArgParser parser("minimap2_options", argparse::default_arguments::none);
    return try_parse_options_impl(parser, minimap2_option_string, error_message);
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

bool print_aln_seq() { return mm_dbg_flag & MM_DBG_PRINT_ALN_SEQ; }

}  // namespace dorado::alignment::mm2