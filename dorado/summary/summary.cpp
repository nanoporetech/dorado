#include "summary.h"

#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/time_utils.h"

#include <spdlog/spdlog.h>

#include <cctype>
#include <csignal>
#include <filesystem>
#include <string_view>

namespace {

class SigIntHandler {
public:
    SigIntHandler() {
#ifndef _WIN32
        std::signal(SIGPIPE, [](int) { interrupt = 1; });
#endif
        std::signal(SIGINT, [](int) { interrupt = 1; });
    }

    ~SigIntHandler() {
#ifndef _WIN32
        std::signal(SIGPIPE, SIG_DFL);
#endif
        std::signal(SIGINT, SIG_DFL);
    }
    static volatile ::sig_atomic_t interrupt;
};

volatile sig_atomic_t SigIntHandler::interrupt{};

}  // anonymous namespace

namespace dorado {

namespace {

using namespace std::string_view_literals;

const std::array s_required_fields = {
        "filename"sv,
        "read_id"sv,
};

const std::array s_general_fields = {
        "run_id"sv,
        "channel"sv,
        "mux"sv,
        "start_time"sv,
        "duration"sv,
        "template_start"sv,
        "template_duration"sv,
        "sequence_length_template"sv,
        "mean_qscore_template"sv,
};

const std::array s_barcoding_fields = {
        "barcode"sv,
};

const std::array s_alignment_fields = {
        "alignment_genome"sv,        "alignment_genome_start"sv,
        "alignment_genome_end"sv,    "alignment_strand_start"sv,
        "alignment_strand_end"sv,    "alignment_direction"sv,
        "alignment_length"sv,        "alignment_num_aligned"sv,
        "alignment_num_correct"sv,   "alignment_num_insertions"sv,
        "alignment_num_deletions"sv, "alignment_num_substitutions"sv,
        "alignment_mapq"sv,          "alignment_strand_coverage"sv,
        "alignment_identity"sv,      "alignment_accuracy"sv,
        "alignment_bed_hits"sv,
};

}  // namespace

SummaryData::SummaryData() = default;

SummaryData::SummaryData(FieldFlags flags) { set_fields(flags); }

void SummaryData::set_separator(char s) { m_separator = s; }

void SummaryData::set_fields(FieldFlags flags) {
    if (flags == 0 || flags > (GENERAL_FIELDS | BARCODING_FIELDS | ALIGNMENT_FIELDS)) {
        throw std::runtime_error(
                "Invalid value of flags option in SummaryData::set_fields method.");
    }
    m_field_flags = flags;
}

void SummaryData::process_file(const std::string& filename, std::ostream& writer) {
    SigIntHandler sig_handler;
    HtsReader reader(filename, std::nullopt);
    m_field_flags = GENERAL_FIELDS | BARCODING_FIELDS;
    if (reader.is_aligned) {
        m_field_flags |= ALIGNMENT_FIELDS;
    }
    auto read_group_exp_start_time = utils::get_read_group_info(reader.header(), "DT");
    write_header(writer);
    write_rows_from_reader(reader, writer, read_group_exp_start_time);
}

bool SummaryData::process_tree(const std::string& folder, std::ostream& writer) {
    std::vector<std::string> files;
    for (const auto& p : utils::fetch_directory_entries(folder, true)) {
        auto ext = std::filesystem::path(p).extension().string();
        if (ext == ".fastq" || ext == ".fq" || ext == ".sam" || ext == ".bam") {
            files.push_back(std::filesystem::absolute(p).string());
        }
    }
    if (files.empty()) {
        spdlog::error("No HTS files found to process.");
        return false;
    }
    SigIntHandler sig_handler;
    write_header(writer);
    for (const auto& read_file : files) {
        HtsReader reader(read_file, std::nullopt);
        auto read_group_exp_start_time = utils::get_read_group_info(reader.header(), "DT");
        write_rows_from_reader(reader, writer, read_group_exp_start_time);
    }
    return true;
}

void SummaryData::write_header(std::ostream& writer) {
    for (size_t i = 0; i < s_required_fields.size(); ++i) {
        if (i > 0) {
            writer << m_separator;
        }
        writer << s_required_fields[i];
    }
    if (m_field_flags & GENERAL_FIELDS) {
        for (size_t i = 0; i < s_general_fields.size(); ++i) {
            writer << m_separator << s_general_fields[i];
        }
    }
    if (m_field_flags & BARCODING_FIELDS) {
        for (size_t i = 0; i < s_barcoding_fields.size(); ++i) {
            writer << m_separator << s_barcoding_fields[i];
        }
    }
    if (m_field_flags & ALIGNMENT_FIELDS) {
        for (size_t i = 0; i < s_alignment_fields.size(); ++i) {
            writer << m_separator << s_alignment_fields[i];
        }
    }
    writer << '\n';
}

void SummaryData::write_rows_from_reader(
        HtsReader& reader,
        std::ostream& writer,
        const std::map<std::string, std::string>& read_group_exp_start_time) {
    while (reader.read() && !SigIntHandler::interrupt) {
        if (reader.record->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
            continue;
        }

        std::string run_id = "unknown";
        std::string model = "unknown";

        auto rg_value = reader.get_tag<std::string>("RG");
        if (rg_value.length() > 0) {
            auto rg_split = rg_value.find('_');
            run_id = rg_value.substr(0, rg_split);
            model = rg_value.substr(rg_split + 1, rg_value.length());
        }

        auto filename = reader.get_tag<std::string>("f5");
        if (filename.empty()) {
            filename = reader.get_tag<std::string>("fn");
        }
        auto read_id = bam_get_qname(reader.record);
        auto channel = reader.get_tag<int>("ch");
        auto mux = reader.get_tag<int>("mx");

        auto start_time_dt = reader.get_tag<std::string>("st");
        auto duration = reader.get_tag<float>("du");

        auto seqlen = reader.record->core.l_qseq;
        auto mean_qscore = reader.get_tag<float>("qs");

        auto num_samples = reader.get_tag<int>("ns");
        auto trim_samples = reader.get_tag<int>("ts");

        auto barcode = reader.get_tag<std::string>("BC");
        if (barcode.empty()) {
            barcode = UNCLASSIFIED;
        }

        float template_duration = duration;
        if (num_samples > 0 && duration > 0) {
            // If either num_samples or duration are 0 (due to missing tags), then
            // we can't properly compute template_duration.
            float sample_rate = num_samples / duration;
            template_duration = (num_samples - trim_samples) / sample_rate;
        }
        auto start_time = 0.0;
        auto exp_start_time_iter = read_group_exp_start_time.find(rg_value);
        if (exp_start_time_iter != read_group_exp_start_time.end()) {
            auto exp_start_dt = exp_start_time_iter->second;
            start_time = utils::time_difference_seconds(start_time_dt, exp_start_dt);
        }
        auto template_start_time = start_time + (duration - template_duration);

        writer << filename << m_separator << read_id;

        if (m_field_flags & GENERAL_FIELDS) {
            writer << m_separator << run_id << m_separator << channel << m_separator << mux
                   << m_separator << start_time << m_separator << duration << m_separator
                   << template_start_time << m_separator << template_duration << m_separator
                   << seqlen << m_separator << mean_qscore;
        }

        if (m_field_flags & BARCODING_FIELDS) {
            writer << m_separator << barcode;
        }

        if (m_field_flags & ALIGNMENT_FIELDS) {
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
            int alignment_bed_hits = 0;

            if (reader.is_aligned && !(reader.record->core.flag & BAM_FUNMAP)) {
                alignment_mapq = static_cast<int>(reader.record->core.qual);
                alignment_genome = reader.header()->target_name[reader.record->core.tid];

                alignment_genome_start = int32_t(reader.record->core.pos);
                alignment_genome_end = int32_t(bam_endpos(reader.record.get()));
                alignment_direction = bam_is_rev(reader.record) ? "-" : "+";

                auto alignment_counts = utils::get_alignment_op_counts(reader.record.get());
                alignment_num_aligned = int(alignment_counts.matches);
                alignment_num_correct =
                        int(alignment_counts.matches - alignment_counts.substitutions);
                alignment_num_insertions = int(alignment_counts.insertions);
                alignment_num_deletions = int(alignment_counts.deletions);
                alignment_num_substitutions = int(alignment_counts.substitutions);
                alignment_length = int(alignment_counts.matches + alignment_counts.insertions +
                                       alignment_counts.deletions);
                alignment_strand_start = int(alignment_counts.softclip_start);
                alignment_strand_end = int(seqlen - alignment_counts.softclip_end);

                strand_coverage = (alignment_strand_end - alignment_strand_start) /
                                  static_cast<float>(seqlen);
                alignment_identity =
                        alignment_num_correct / static_cast<float>(alignment_counts.matches);
                alignment_accurary = alignment_num_correct / static_cast<float>(alignment_length);
                alignment_bed_hits = reader.get_tag<int>("bh");
            }

            writer << m_separator << alignment_genome << m_separator << alignment_genome_start
                   << m_separator << alignment_genome_end << m_separator << alignment_strand_start
                   << m_separator << alignment_strand_end << m_separator << alignment_direction
                   << m_separator << alignment_length << m_separator << alignment_num_aligned
                   << m_separator << alignment_num_correct << m_separator
                   << alignment_num_insertions << m_separator << alignment_num_deletions
                   << m_separator << alignment_num_substitutions << m_separator << alignment_mapq
                   << m_separator << strand_coverage << m_separator << alignment_identity
                   << m_separator << alignment_accurary << m_separator << alignment_bed_hits;
            ;
        }
        writer << '\n';
    }
}

}  // namespace dorado
