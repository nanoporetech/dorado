#include "hts_writer/SummaryFileWriter.h"

#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "utils/barcode_kits.h"
#include "utils/string_utils.h"

#include <htslib/sam.h>

#include <array>
#include <string_view>
#include <vector>

namespace {
using namespace std::string_view_literals;

const std::string_view separator = "\t";

const std::array s_general_fields = {
        "input_filename"sv, "batch_id"sv, "parent_read_id"sv, "read_id"sv,    "run_id"sv,
        "channel"sv,        "mux"sv,      "minknow_events"sv, "start_time"sv, "duration"sv,
};

const std::array s_basecalling_fields = {
        "passes_filtering"sv,  "template_start"sv,           "num_events_template"sv,
        "template_duration"sv, "sequence_length_template"sv, "mean_qscore_template"sv,
};

const std::array s_polya_fields = {
        "poly_tail_length"sv, "poly_tail_start"sv, "poly_tail_end"sv,
        "poly_tail2_start"sv, "poly_tail2_end"sv,
};

const std::array s_experiment_fields = {
        "pore_type"sv,
        "experiment_id"sv,
        "sample_id"sv,
        "end_reason"sv,
};

const std::array s_barcoding_fields = {
        "alias"sv,
        "type"sv,
        "barcode_arrangement"sv,
        "barcode_kit"sv,
        "barcode_variant"sv,
        "barcode_score"sv,
        "barcode_front_score"sv,
        "barcode_front_foundseq_length"sv,
        "barcode_front_begin_index"sv,
        "barcode_rear_score"sv,
        "barcode_rear_foundseq_length"sv,
        "barcode_rear_end_index"sv,
};

const std::array s_alignment_fields = {
        "alignment_genome"sv,
        "alignment_direction"sv,
        "alignment_genome_start"sv,
        "alignment_genome_end"sv,
        "alignment_strand_start"sv,
        "alignment_strand_end"sv,
        "alignment_num_insertions"sv,
        "alignment_num_deletions"sv,
        "alignment_num_aligned"sv,
        "alignment_num_correct"sv,
        "alignment_identity"sv,
        "alignment_accuracy"sv,
        "alignment_score"sv,
        "alignment_coverage"sv,
        "alignment_bed_hits"sv,
        "alignment_mapping_quality"sv,
        "alignment_num_alignments"sv,
        "alignment_num_secondary_alignments"sv,
        "alignment_num_supplementary_alignments"sv,
};

const std::array s_duplex_fields = {
        "duplex_parent_template"sv,
        "duplex_parent_complement"sv,
};

template <typename T>
T get_tag(bam1_t* record, const char* tagname, const T& default_value) {
    T tag_value{};
    uint8_t* tag = bam_aux_get(record, tagname);

    if (!tag) {
        return default_value;
    }
    if constexpr (std::is_integral_v<T>) {
        tag_value = static_cast<T>(bam_aux2i(tag));
    } else if constexpr (std::is_floating_point_v<T>) {
        tag_value = static_cast<T>(bam_aux2f(tag));
    } else {
        const char* val = bam_aux2Z(tag);
        tag_value = val ? val : default_value;
    }

    return tag_value;
}

template <typename T>
std::vector<T> get_array(bam1_t* record, const char* tagname) {
    uint8_t* tag = bam_aux_get(record, tagname);
    if (!tag) {
        return {};
    }

    uint32_t len = bam_auxB_len(tag);
    std::vector<T> tag_value;
    tag_value.reserve(len);
    for (uint32_t idx = 0; idx < len; ++idx) {
        if constexpr (std::is_integral_v<T>) {
            tag_value.push_back(static_cast<T>(bam_auxB2i(tag, idx)));
        } else if constexpr (std::is_floating_point_v<T>) {
            tag_value.push_back(static_cast<T>(bam_auxB2f(tag, idx)));
        } else {
            throw std::logic_error("Invalid type for array tag.");
        }
    }
    return tag_value;
}

}  // namespace

namespace dorado::hts_writer {

SummaryFileWriter::SummaryFileWriter(const std::filesystem::path& output_directory,
                                     FieldFlags flags)
        : m_field_flags(flags),
          m_summary_file(output_directory / "sequencing_summary.txt"),
          m_summary_stream(m_summary_file) {
    init();
}

SummaryFileWriter::SummaryFileWriter(std::ostream& stream, FieldFlags flags)
        : m_field_flags(flags), m_summary_stream(stream) {
    init();
}

void SummaryFileWriter::init() {
    // Write column headers
    for (size_t i = 0; i < s_general_fields.size(); ++i) {
        if (i > 0) {
            m_summary_stream << separator;
        }
        m_summary_stream << s_general_fields[i];
    }
    if (m_field_flags & BASECALLING_FIELDS) {
        for (size_t i = 0; i < s_basecalling_fields.size(); ++i) {
            m_summary_stream << separator << s_basecalling_fields[i];
        }
    }
    if (m_field_flags & POLYA_FIELDS) {
        for (size_t i = 0; i < s_polya_fields.size(); ++i) {
            m_summary_stream << separator << s_polya_fields[i];
        }
    }
    if (m_field_flags & EXPERIMENT_FIELDS) {
        for (size_t i = 0; i < s_experiment_fields.size(); ++i) {
            m_summary_stream << separator << s_experiment_fields[i];
        }
    }
    if (m_field_flags & BARCODING_FIELDS) {
        for (size_t i = 0; i < s_barcoding_fields.size(); ++i) {
            m_summary_stream << separator << s_barcoding_fields[i];
        }
    }
    if (m_field_flags & ALIGNMENT_FIELDS) {
        for (size_t i = 0; i < s_alignment_fields.size(); ++i) {
            m_summary_stream << separator << s_alignment_fields[i];
        }
    }
    if (m_field_flags & DUPLEX_FIELDS) {
        for (size_t i = 0; i < s_duplex_fields.size(); ++i) {
            m_summary_stream << separator << s_duplex_fields[i];
        }
    }

    m_summary_stream << '\n';
}

void SummaryFileWriter::process(const Processable item) {
    dispatch_processable(item, [this](const auto& t) { this->handle(t); });
}

void SummaryFileWriter::handle(const HtsData& data) {
    // skip secondary and supplementary alignments in the summary
    if (data.bam_ptr->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
        return;
    }

    // Write column data

    auto record = data.bam_ptr.get();
    m_summary_stream << get_tag<std::string>(record, "fn", "unknown");
    m_summary_stream << separator << "0";  // batch_id
    m_summary_stream << separator << get_tag<std::string>(record, "pi", bam_get_qname(record));
    m_summary_stream << separator << bam_get_qname(record);
    m_summary_stream << separator << data.read_attrs.protocol_run_id;
    m_summary_stream << separator << get_tag(record, "ch", 0);
    m_summary_stream << separator << get_tag(record, "mx", 0);
    m_summary_stream << separator << data.read_attrs.num_minknow_events;
    m_summary_stream << separator << (data.read_attrs.start_time_ms / 1000.f);
    m_summary_stream << separator << get_tag(record, "du", 0.f);

    if (m_field_flags & BASECALLING_FIELDS) {
        m_summary_stream << separator << (data.read_attrs.is_status_pass ? "TRUE" : "FALSE");
        m_summary_stream << separator << 0;    // template_start - calculate?
        m_summary_stream << separator << 0;    // num_events_template
        m_summary_stream << separator << 0.f;  // template_duration
        m_summary_stream << separator << record->core.l_qseq;
        m_summary_stream << separator << get_tag(record, "qs", 0.f);
    }
    if (m_field_flags & POLYA_FIELDS) {
        m_summary_stream << separator << get_tag(record, "pt", -2);
        std::array<int, 4> polya_stats;
        polya_stats.fill(-1);
        auto stats = get_array<int>(record, "pa");
        if (!stats.empty()) {
            // skip the anchor info, we only output the ranges
            std::copy_n(std::next(std::begin(stats)), std::size(polya_stats),
                        std::begin(polya_stats));
        }
        for (const auto& stat : polya_stats) {
            m_summary_stream << separator << stat;
        }
    }
    if (m_field_flags & EXPERIMENT_FIELDS) {
        m_summary_stream << separator << data.read_attrs.pore_type;
        m_summary_stream << separator << data.read_attrs.experiment_id;
        m_summary_stream << separator << data.read_attrs.sample_id;
        m_summary_stream << separator << data.read_attrs.end_reason;
    }
    if (m_field_flags & BARCODING_FIELDS) {
        std::string alias = "unknown";
        std::string type = "unknown";
        std::string barcode_arrangement = "unknown";
        std::string barcode_kit = "unknown";
        std::string barcode_variant = "unknown";
        float barcode_score = -1.f;
        float barcode_front_score = -1.f;
        int barcode_front_foundseq_length = -1;
        int barcode_front_begin_index = -1;
        float barcode_rear_score = -1.f;
        int barcode_rear_foundseq_length = -1;
        int barcode_rear_end_index = -1;

        if (data.barcoding_result) {
            // retrieve data from barcoding result
            // alias = ???
            // type = ???
            barcode_arrangement = data.barcoding_result->barcode_name;
            barcode_kit = data.barcoding_result->barcode_kit;
            barcode_variant = data.barcoding_result->variant;
            barcode_score = data.barcoding_result->barcode_score;
            barcode_front_score = data.barcoding_result->top_barcode_score;
            barcode_rear_score = data.barcoding_result->bottom_barcode_score;
            barcode_front_foundseq_length = data.barcoding_result->top_barcode_pos.second -
                                            data.barcoding_result->top_barcode_pos.first;
            barcode_rear_foundseq_length = data.barcoding_result->bottom_barcode_pos.second -
                                           data.barcoding_result->bottom_barcode_pos.first;
            barcode_front_begin_index = data.barcoding_result->top_barcode_pos.first;
            barcode_rear_end_index = data.barcoding_result->bottom_barcode_pos.second;
        }

        m_summary_stream << separator << alias;
        m_summary_stream << separator << type;
        m_summary_stream << separator << barcode_arrangement;
        m_summary_stream << separator << barcode_kit;
        m_summary_stream << separator << barcode_variant;
        m_summary_stream << separator << barcode_score;
        m_summary_stream << separator << barcode_front_score;
        m_summary_stream << separator << barcode_front_foundseq_length;
        m_summary_stream << separator << barcode_front_begin_index;
        m_summary_stream << separator << barcode_rear_score;
        m_summary_stream << separator << barcode_rear_foundseq_length;
        m_summary_stream << separator << barcode_rear_end_index;
    }
    if (m_field_flags & ALIGNMENT_FIELDS) {
        std::string alignment_genome = "*";
        int32_t alignment_genome_start = -1;
        int32_t alignment_genome_end = -1;
        int32_t alignment_strand_start = -1;
        int32_t alignment_strand_end = -1;
        std::string alignment_direction = "*";
        int32_t alignment_length = 0;
        int32_t alignment_mapping_quality = 0;
        int alignment_num_aligned = 0;
        int alignment_num_correct = 0;
        int alignment_num_insertions = 0;
        int alignment_num_deletions = 0;
        float alignment_coverage = 0.0;
        float alignment_identity = 0.0;
        float alignment_accuracy = 0.0;
        int alignment_bed_hits = 0;
        int alignment_score = 0;
        int alignment_num_alignments = 0;
        int alignment_num_secondary_alignments = 0;
        int alignment_num_supplementary_alignments = 0;

        if (!(record->core.flag & BAM_FUNMAP)) {
            // alignment_genome = get genome from header
            alignment_direction = bam_is_rev(record) ? "-" : "+";
            alignment_genome_start = int32_t(record->core.pos);
            alignment_genome_end = int32_t(bam_endpos(record));

            auto alignment_counts = utils::get_alignment_op_counts(record);
            alignment_strand_start = int(alignment_counts.softclip_start);
            alignment_strand_end = int(record->core.l_qseq - alignment_counts.softclip_end);
            alignment_num_insertions = int(alignment_counts.insertions);
            alignment_num_deletions = int(alignment_counts.deletions);
            alignment_num_aligned = int(alignment_counts.matches);
            alignment_num_correct = int(alignment_counts.matches - alignment_counts.substitutions);
            alignment_identity =
                    alignment_num_correct / static_cast<float>(alignment_counts.matches);
            alignment_length = int(alignment_counts.matches + alignment_counts.insertions +
                                   alignment_counts.deletions);
            alignment_accuracy = alignment_num_correct / static_cast<float>(alignment_length);
            alignment_score = get_tag(record, "AS", 0);

            alignment_coverage = (alignment_strand_end - alignment_strand_start) /
                                 static_cast<float>(record->core.l_qseq);
            alignment_bed_hits = get_tag(record, "bh", 0);
            alignment_mapping_quality = record->core.qual;
            alignment_num_alignments = 0;                // set in read attributes
            alignment_num_secondary_alignments = 0;      // set in read attributes
            alignment_num_supplementary_alignments = 0;  // set in read attributes
        }
        m_summary_stream << separator << alignment_genome;
        m_summary_stream << separator << alignment_direction;
        m_summary_stream << separator << alignment_genome_start;
        m_summary_stream << separator << alignment_genome_end;
        m_summary_stream << separator << alignment_strand_start;
        m_summary_stream << separator << alignment_strand_end;
        m_summary_stream << separator << alignment_num_insertions;
        m_summary_stream << separator << alignment_num_deletions;
        m_summary_stream << separator << alignment_num_aligned;
        m_summary_stream << separator << alignment_num_correct;
        m_summary_stream << separator << alignment_identity;
        m_summary_stream << separator << alignment_accuracy;
        m_summary_stream << separator << alignment_score;
        m_summary_stream << separator << alignment_coverage;
        m_summary_stream << separator << alignment_bed_hits;
        m_summary_stream << separator << alignment_mapping_quality;
        m_summary_stream << separator << alignment_num_alignments;
        m_summary_stream << separator << alignment_num_secondary_alignments;
        m_summary_stream << separator << alignment_num_supplementary_alignments;
    }
    if (m_field_flags & DUPLEX_FIELDS) {
        auto read_ids = utils::split(bam_get_qname(record), ';');
        std::string_view duplex_parent_template = "-"sv;
        std::string_view duplex_parent_complement = "-"sv;
        if (read_ids.size() == 2) {
            duplex_parent_template = read_ids[0];
            duplex_parent_complement = read_ids[1];
        }
        m_summary_stream << separator << duplex_parent_template;
        m_summary_stream << separator << duplex_parent_complement;
    }

    m_summary_stream << '\n';
}

void SummaryFileWriter::shutdown() {
    // close the file if we're working with a file path
    if (m_summary_file.is_open()) {
        m_summary_file.close();
    }
}

}  // namespace dorado::hts_writer
