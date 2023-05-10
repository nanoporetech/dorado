#include "Version.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <cctype>
#include <csignal>
#include <filesystem>

namespace dorado {

volatile sig_atomic_t interupt = 0;

using HtsReader = utils::HtsReader;

std::vector<std::string> header = {
        "filename",
        "read_id",
        "run_id",
        "channel",
        "mux",
        "start_time",
        "duration",
        "template_start",
        "template_duration",
        "sequence_length_template",
        "mean_qscore_template",
};

std::vector<std::string> aligned_header = {
        "alignment_genome",         "alignment_genome_start",    "alignment_genome_end",
        "alignment_strand_start",   "alignment_strand_end",      "alignment_direction",
        "alignment_length",         "alignment_num_aligned",     "alignment_num_correct",
        "alignment_num_insertions", "alignment_num_deletions",   "alignment_num_substitutions",
        "alignment_mapq",           "alignment_strand_coverage", "alignment_identity",
        "alignment_accuracy"};

int summary(int argc, char *argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("reads").help("SAM/BAM file produced by dorado basecaller.");
    parser.add_argument("-s", "--separator").default_value(std::string("\t"));
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception &e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto reads(parser.get<std::string>("reads"));
    auto separator(parser.get<std::string>("separator"));

    HtsReader reader(reads);
    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
#ifndef _WIN32
    std::signal(SIGPIPE, [](int signum) { interupt = 1; });
#endif
    std::signal(SIGINT, [](int signum) { interupt = 1; });

    // HEADER
    for (int col = 0; col < header.size() - 1; col++) {
        std::cout << header[col] << separator;
    }
    std::cout << header[header.size() - 1];

    if (reader.is_aligned) {
        for (int col = 0; col < aligned_header.size() - 1; col++) {
            std::cout << separator << aligned_header[col];
        }
        std::cout << separator << aligned_header[aligned_header.size() - 1];
    }

    std::cout << '\n';

    while (reader.read() && !interupt) {
        if (reader.record->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
            continue;
        }

        auto rg_value = reader.get_tag<std::string>("RG");
        auto filename = reader.get_tag<std::string>("f5");
        if (filename.empty()) {
            filename = reader.get_tag<std::string>("fn");
        }
        auto read_id = bam_get_qname(reader.record);
        auto channel = reader.get_tag<int>("ch");
        auto mux = reader.get_tag<int>("mx");
        auto start_time = reader.get_tag<std::string>("st");
        auto duration = reader.get_tag<float>("du");
        auto seqlen = reader.record->core.l_qseq;
        auto mean_qscore = reader.get_tag<int>("qs");

        auto num_samples = reader.get_tag<int>("ns");
        auto trim_samples = reader.get_tag<int>("ts");

        // todo: sample_rate and template_start_time
        float template_duration = (num_samples - trim_samples) / 4000.0f;
        auto template_start_time = start_time;

        std::cout << filename << separator << read_id << separator << rg_value.substr(0, 36)
                  << separator << channel << separator << mux << separator << start_time
                  << separator << duration << separator << template_start_time << separator
                  << template_duration << separator << seqlen << separator << mean_qscore;

        int32_t query_start = 0;
        int32_t query_end = 0;
        std::string alignment_genome = "*";
        int32_t alignment_genome_start = -1;
        int32_t alignment_genome_end = -1;
        int32_t alignment_strand_start = -1;
        int32_t alignment_strand_end = -1;
        std::string alignment_direction = "*";
        int32_t alignment_length = 0;
        int32_t alignment_mapq = 0;
        int alignment_num_aligned = 0;
        int alignment_num_correct = 0;
        int alignment_num_insertions = 0;
        int alignment_num_deletions = 0;
        int alignment_num_substitutions = 0;
        float strand_coverage = 0.0;
        float alignment_identity = 0.0;
        float alignment_accurary = 0.0;

        if (!(reader.record->core.flag & BAM_FUNMAP)) {
            alignment_mapq = static_cast<int>(reader.record->core.qual);
            alignment_genome = reader.header->target_name[reader.record->core.tid];

            alignment_genome_start = reader.record->core.pos;
            alignment_genome_end = bam_endpos(reader.record.get());

            alignment_strand_start = 0;
            alignment_strand_end = seqlen;

            alignment_direction = bam_is_rev(reader.record) ? "-" : "+";
            alignment_length = reader.record->core.l_qseq;

            uint32_t *cigar = bam_get_cigar(reader.record);
            int n_cigar = reader.record->core.n_cigar;

            if (bam_cigar_op(cigar[0]) == BAM_CSOFT_CLIP) {
                alignment_strand_start += bam_cigar_oplen(cigar[0]);
            }

            if (bam_cigar_op(cigar[n_cigar - 1]) == BAM_CSOFT_CLIP) {
                alignment_strand_end -= bam_cigar_oplen(cigar[n_cigar - 1]);
            }

            for (int i = 0; i < n_cigar; ++i) {
                int op = bam_cigar_op(cigar[i]);
                int op_len = bam_cigar_oplen(cigar[i]);

                switch (op) {
                case BAM_CMATCH:
                    alignment_num_aligned += op_len;
                    break;
                case BAM_CINS:
                    alignment_num_insertions += op_len;
                    break;
                case BAM_CDEL:
                    alignment_num_deletions += op_len;
                    break;
                default:
                    break;
                }
            }

            uint8_t *md_ptr = bam_aux_get(reader.record.get(), "MD");

            if (md_ptr) {
                char *md = bam_aux2Z(md_ptr);

                int md_length = 0;
                int i = 0;
                while (md[i]) {
                    if (std::isdigit(md[i])) {
                        md_length = md_length * 10 + (md[i] - '0');
                    } else {
                        if (md[i] == '^') {
                            // Skip deletions
                            i++;
                            while (md[i] && !std::isdigit(md[i])) {
                                i++;
                            }
                        } else {
                            // Substitution found
                            alignment_num_substitutions++;
                            md_length++;
                        }
                    }
                    i++;
                }
            }
            alignment_num_correct = alignment_num_aligned - alignment_num_substitutions;

            alignment_identity = alignment_num_correct / static_cast<float>(alignment_num_aligned);
            alignment_accurary = alignment_num_correct /
                                 static_cast<float>(alignment_genome_end - alignment_genome_start);
            strand_coverage =
                    (alignment_strand_end - alignment_strand_start) / static_cast<float>(seqlen);
        }

        std::cout << separator << alignment_genome << separator << alignment_genome_start
                  << separator << alignment_genome_end << separator << alignment_strand_start
                  << separator << alignment_strand_end << separator << alignment_direction
                  << separator << alignment_genome_end - alignment_genome_start << separator
                  << alignment_num_aligned << separator << alignment_num_correct << separator
                  << alignment_num_insertions << separator << alignment_num_deletions << separator
                  << alignment_num_substitutions << separator << alignment_mapq << separator
                  << strand_coverage << separator << alignment_identity << separator
                  << alignment_accurary;

        std::cout << '\n';
    }

    return 0;
}

}  // namespace dorado
