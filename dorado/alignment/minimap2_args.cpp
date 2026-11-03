#include "alignment/minimap2_args.h"

#include "alignment/minimap2_wrappers.h"  // IWYU pragma: keep for Minimap2IdxOptHolder
#include "utils/arg_parse_ext.h"
#include "utils/log_utils.h"
#include "utils/string_utils.h"

#include <argparse/argparse.hpp>
#include <minimap.h>
#include <mmpriv.h>

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace {

constexpr inline std::string_view MM2_OPTS_ARG = "--mm2-opts";

void mm_mapopt_override(mm_mapopt_t* mapopt) {
    // Force cigar generation.
    mapopt->flag |= MM_F_CIGAR;

    // Always emit MD tags.
    mapopt->flag |= MM_F_OUT_MD;

    // Equivalent to "--cap-kalloc 100m --cap-sw-mem 50m"
    mapopt->cap_kalloc = 100'000'000;
    mapopt->max_sw_mat = 50'000'000;
}

void mm_idxopt_override(mm_idxopt_t* idxopt) {
    idxopt->batch_size = 16000000000;
    idxopt->mini_batch_size = idxopt->batch_size;
}

void mm_mapopt_set_defaults(mm_mapopt_t& mapopt) {
    mm_mapopt_init(&mapopt);
    mm_mapopt_override(&mapopt);
}

void mm_idxopt_set_defaults(mm_idxopt_t& idxopt) {
    mm_idxopt_init(&idxopt);
    mm_idxopt_override(&idxopt);
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

void add_options_string_arg(argparse::ArgumentParser& parser) {
    parser.add_argument(MM2_OPTS_ARG)
            .help("Optional minimap2 options string. For multiple arguments surround with double "
                  "quotes.");
}

namespace {

void add_arguments(argparse::ArgumentParser& parser) {
    parser.add_argument("-k")
            .help("minimap2 k-mer size for alignment (maximum 28).")
            .template scan<'i', int>();

    parser.add_argument("-w")
            .help("minimap2 minimizer window size for alignment.")
            .template scan<'i', int>();

    parser.add_argument("-I").help("minimap2 index batch size.");

    parser.add_argument("--secondary").help("minimap2 outputs secondary alignments");

    parser.add_argument("-N")
            .help("minimap2 retains at most INT secondary alignments")
            .template scan<'i', int>();

    parser.add_argument("-Y")
            .help("minimap2 uses soft clipping for supplementary alignments")
            .flag();

    parser.add_argument("-r").help(
            "minimap2 chaining/alignment bandwidth and optionally long-join bandwidth "
            "specified as NUM,[NUM]");

    parser.add_argument("--junc-bed")
            .help("optional file with gene annotations in the BED12 format (aka 12-column BED), or "
                  "intron positions in 5-column BED. With this option, minimap2 prefers splicing "
                  "in annotations.");
    // Setting options to lr:hq which is appropriate for high quality nanopore reads.
    parser.add_argument("-x")
            .help("minimap2 preset for indexing and mapping.")
            .default_value(std::string{DEFAULT_MM_PRESET})
            .nargs(1);

    parser.add_argument("--eqx").help("write =/X CIGAR operators").flag();
    parser.add_argument("--MD").help("output the MD tag (no-op since we do by default)").flag();
    parser.add_argument("--cs")
            .help("output the cs tag")
            .choices("none", "short", "long")
            .default_value("none")
            .nargs(1);
    parser.add_argument("--rmq").help("use the minigraph chaining algorithm").flag();

    parser.add_argument("-u")
            .help("how to find GT-AG. "
                  "f:transcript strand, "
                  "r:reverse strand (match CT-AC), "
                  "b:both strands, "
                  "n:don't match GT-AG")
            .choices("f", "r", "b", "n")
            .default_value("n")
            .nargs(1);

    parser.add_argument("--secondary-seq")
            .hidden()
            .help("minimap2 output seq/qual for secondary and supplementary alignments")
            .flag();

    parser.add_argument("--print-aln-seq")
            .hidden()
            .help("minimap2 debug print qname and aln_seq")
            .flag();
}

void apply_preset(Minimap2Options& options, const std::string& preset) {
    if (preset.empty() || mm_set_opt(preset.c_str(), &options.index.index_options->get(),
                                     &options.mapping.mapping_options->get()) == 0) {
        return;
    }
    throw std::runtime_error("Cannot set mm2 options with preset: " + preset);
}

void apply_indexing_options(const argparse::ArgumentParser& parser, mm_idxopt_t& options) {
    auto kmer = get_optional_as<short>(parser.present<int>("k"));
    if (kmer) {
        options.k = *kmer;
    }
    auto window_size = get_optional_as<short>(parser.present<int>("w"));
    if (window_size) {
        options.w = *window_size;
    }
    auto index_batch_size = parser.present<std::string>("I");
    if (index_batch_size) {
        options.batch_size = utils::arg_parse::parse_string_to_size<uint64_t>(*index_batch_size);
    }
    options.mini_batch_size = options.batch_size;
}

void apply_mapping_options(const argparse::ArgumentParser& parser, mm_mapopt_t& options) {
    auto secondary = parser.present<std::string>("--secondary");
    bool print_secondary{true};
    if (secondary && !utils::arg_parse::parse_yes_or_no(*secondary)) {
        print_secondary = false;
    }

    auto best_n_secondary = parser.present<int>("N");
    if (best_n_secondary.value_or(1) == 0) {
        DORADO_LOG(warn, "Ignoring '-N 0', using preset default");
        print_secondary = true;
    } else {
        options.best_n = best_n_secondary.value_or(options.best_n);
    }

    if (!print_secondary) {
        options.flag |= MM_F_NO_PRINT_2ND;
    }

    auto optional_bandwidth = parser.present<std::string>("-r");
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
    if (parser.get<bool>("Y")) {
        options.flag |= MM_F_SOFTCLIP;
    }
    if (parser.get<bool>("secondary-seq")) {
        options.flag |= MM_F_SECONDARY_SEQ;
    }
    if (parser.get<bool>("eqx")) {
        options.flag |= MM_F_EQX;
    }
    if (parser.get<bool>("rmq")) {
        options.flag |= MM_F_RMQ;
    }
}

std::optional<Minimap2Options> process_arguments(const argparse::ArgumentParser& parser,
                                                 std::string& error_message) {
    Minimap2Options res;
    mm_idxopt_set_defaults(res.index.index_options->get());
    mm_mapopt_set_defaults(res.mapping.mapping_options->get());

    // apply preset before overwriting with other user supplied options.
    apply_preset(res, parser.get<std::string>("-x"));

    apply_indexing_options(parser, res.index.index_options->get());
    apply_mapping_options(parser, res.mapping.mapping_options->get());

    auto rc = mm_check_opt(&res.index.index_options->get(), &res.mapping.mapping_options->get());
    if (rc < 0) {
        error_message = "Invalid minimap2 options string. Error code: " + std::to_string(rc);
        return std::nullopt;
    }

    // Cache the --junc-bed arg with the index options for use when the index is loaded
    auto junc_bed = parser.present<std::string>("--junc-bed");
    if (junc_bed) {
        res.index.junc_bed = std::move(*junc_bed);
    }

    if (parser.get<bool>("print-aln-seq")) {
        // set the global flags
        mm_dbg_flag |= MM_DBG_PRINT_QNAME | MM_DBG_PRINT_ALN_SEQ;
    }

    apply_cs_option(res, parser.get("cs"));
    apply_u_option(res, parser.get("u"));

    return res;
}

}  // namespace

std::string get_help_message() {
    argparse::ArgumentParser parser("minimap2_options", MM_VERSION,
                                    argparse::default_arguments::none);
    add_arguments(parser);
    return parser.help().str();
}

Minimap2Options parse_options(const std::string& minimap2_option_string) {
    std::string error_message{};
    auto minimap2_options = try_parse_options(minimap2_option_string, error_message);
    if (!minimap2_options) {
        DORADO_LOG(error, "{}\n{}", error_message, get_help_message());
        throw std::runtime_error(error_message);
    }
    return *minimap2_options;
}

// Split attached -u values (e.g. "-uf") into a separate "-u" entry followed by the value.
// This type of parameter passing is not supported by argparse, but mm2 allows it.
// This is done specifically for this parameter to allow users to copypasta their mm2 commandline options
// Invalid values are left for the parser to reject.
void split_attached_u_values(std::vector<std::string>& mm2_args) {
    for (size_t i = 0; i < mm2_args.size(); ++i) {
        if (mm2_args[i].size() > 2 && mm2_args[i].starts_with("-u")) {
            std::string value = mm2_args[i].substr(2);
            mm2_args[i].resize(2);
            mm2_args.insert(mm2_args.begin() + i + 1, std::move(value));
        }
    }
}

namespace {

std::optional<Minimap2Options> try_parse_options_impl(argparse::ArgumentParser& parser,
                                                      const std::string& minimap2_option_string,
                                                      std::string& error_message) {
    std::vector<std::string> mm2_args = [&minimap2_option_string] {
        if (minimap2_option_string.empty()) {
            return std::vector<std::string>{"minimap2_options"};
        }
        return utils::split("minimap2_options " + minimap2_option_string, ' ');
    }();
    split_attached_u_values(mm2_args);
    add_arguments(parser);
    try {
        parser.parse_args(mm2_args);
    } catch (const std::exception& e) {
        error_message = e.what();
        return std::nullopt;
    }

    return process_arguments(parser, error_message);
}

}  // namespace

std::optional<Minimap2Options> try_parse_options(const std::string& minimap2_option_string,
                                                 std::string& error_message) {
    argparse::ArgumentParser parser("minimap2_options", MM_VERSION);
    return try_parse_options_impl(parser, minimap2_option_string, error_message);
}

std::optional<Minimap2Options> try_parse_options_no_help(const std::string& minimap2_option_string,
                                                         std::string& error_message) {
    argparse::ArgumentParser parser("minimap2_options", MM_VERSION,
                                    argparse::default_arguments::none);
    return try_parse_options_impl(parser, minimap2_option_string, error_message);
}

void apply_cs_option(Minimap2Options& options, std::string_view cs_opt) {
    if (cs_opt.empty()) {
        return;
    }
    auto& flag = options.mapping.mapping_options->get().flag;

    flag |= MM_F_OUT_CS | MM_F_CIGAR;
    if (cs_opt == "short") {
        flag &= ~MM_F_OUT_CS_LONG;
    } else if (cs_opt == "long") {
        flag |= MM_F_OUT_CS_LONG;
    } else if (cs_opt == "none") {
        flag &= ~MM_F_OUT_CS;
    } else {
        DORADO_LOG(warn, "Unrecognized options for --cs={}", cs_opt);
    }
}

void apply_u_option(Minimap2Options& options, std::string_view u_opt) {
    auto& flag = options.mapping.mapping_options->get().flag;

    if (u_opt == "b") {
        flag |= MM_F_SPLICE_FOR | MM_F_SPLICE_REV;  // both strands
    } else if (u_opt == "f") {
        flag |= MM_F_SPLICE_FOR;
        flag &= ~MM_F_SPLICE_REV;  // match GT-AG
    } else if (u_opt == "r") {
        flag |= MM_F_SPLICE_REV;
        flag &= ~MM_F_SPLICE_FOR;  // match CT-AC (reverse complement of GT-AG)
    } else if (u_opt == "n") {
        flag &= ~(MM_F_SPLICE_FOR | MM_F_SPLICE_REV);  // don't try to match the GT-AG signal
    } else {
        DORADO_LOG(error, "Unrecognized option for -u={}", u_opt);
    }
}

void apply_dual_option(Minimap2Options& options, std::string_view dual) {
    if (dual.empty()) {
        return;
    }

    if (dual == "yes") {
        options.mapping.mapping_options->get().flag &= ~MM_F_NO_DUAL;
    } else if (dual == "no") {
        options.mapping.mapping_options->get().flag |= MM_F_NO_DUAL;
    } else {
        DORADO_LOG(warn, "Unrecognized options for --dual={}", dual);
    }
}

bool print_aln_seq() { return mm_dbg_flag & MM_DBG_PRINT_ALN_SEQ; }

}  // namespace dorado::alignment::mm2