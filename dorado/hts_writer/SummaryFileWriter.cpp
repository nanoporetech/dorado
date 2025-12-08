#include "hts_writer/SummaryFileWriter.h"

#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_writer_utils.h"
#include "utils/barcode_kits.h"
#include "utils/string_utils.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <array>
#include <iomanip>
#include <ostream>
#include <sstream>
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

int get_min_qscore(sam_hdr_t* header) {
    auto command_line_cl =
            dorado::utils::extract_pg_keys_from_hdr(header, {"CL"}, "ID", "basecaller");
    // If dorado was run with --min-qscore option, parse the value so we can re-evaluate the pass/fail criterion
    std::stringstream cl{command_line_cl["CL"]};
    std::string out;
    while (cl.good()) {
        cl >> std::quoted(out);
        if (out == "--min-qscore") {
            cl >> std::quoted(out);
            return std::atoi(out.c_str());
        }
    }
    return 0;
}

std::ofstream create_file_stream(const std::filesystem::path& output_path) {
    dorado::hts_writer::create_output_folder(output_path);
    return std::ofstream(output_path);
}

}  // namespace

namespace dorado::hts_writer {

SummaryFileWriter::SummaryFileWriter(const std::filesystem::path& output_directory,
                                     FieldFlags flags)
        : m_field_flags(flags),
          m_summary_file(create_file_stream(output_directory / "sequencing_summary.txt")),
          m_summary_stream(m_summary_file) {
    init();
}

SummaryFileWriter::SummaryFileWriter(std::ostream& stream, FieldFlags flags)
        : m_field_flags(flags), m_summary_stream(stream) {
    init();
}

void SummaryFileWriter::init() {
    m_summary_stream << std::fixed << std::setprecision(6);

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

void SummaryFileWriter::set_shared_header(dorado::SamHdrSharedPtr header) {
    if (m_dynamic_header != nullptr) {
        throw std::logic_error("set_shared_header is incompatible with set_dynamic_header.");
    }
    m_shared_header = std::move(header);
}

void SummaryFileWriter::set_dynamic_header(
        const std::shared_ptr<utils::HeaderMapper::HeaderMap>& header_map) {
    if (m_shared_header != nullptr) {
        throw std::logic_error("set_dynamic_header is incompatible with set_shared_header.");
    }
    m_dynamic_header = header_map;
};

void SummaryFileWriter::process(const Processable item) {
    dispatch_processable(item, [this](const auto& t) { this->handle(t); });
}

void SummaryFileWriter::handle(const HtsData& data) const {
    // skip secondary and supplementary alignments in the summary
    if (data.bam_ptr->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) {
        return;
    }

    // Write column data

    auto record = data.bam_ptr.get();
    auto duration = get_tag(record, "du", 0.f);
    auto num_samples = get_tag(record, "ns", 0);
    auto sample_rate = (duration > 0) ? num_samples / duration : 0.f;

    m_summary_stream << get_tag<std::string>(record, "fn", "unknown");
    m_summary_stream << separator << "0";  // batch_id
    m_summary_stream << separator << get_tag<std::string>(record, "pi", bam_get_qname(record));
    m_summary_stream << separator << bam_get_qname(record);
    m_summary_stream << separator << data.read_attrs.protocol_run_id;
    m_summary_stream << separator << get_tag(record, "ch", 0);
    m_summary_stream << separator << get_tag(record, "mx", 0);
    m_summary_stream << separator << get_tag(record, "me", uint32_t(0));
    m_summary_stream << separator << (data.read_attrs.start_time_ms / 1000.f);
    m_summary_stream << separator << duration;

    if (m_field_flags & BASECALLING_FIELDS) {
        auto template_start = sample_rate != 0 ? (data.read_attrs.start_time_ms / 1000.f) +
                                                         (get_tag(record, "ts", 0) / sample_rate)
                                               : 0.f;
        auto template_samples = get_tag(record, "ns", 0) - get_tag(record, "ts", 0);
        auto template_duration = sample_rate != 0 ? float(template_samples) / sample_rate : 0.f;

        m_summary_stream << separator << (data.read_attrs.is_status_pass ? "TRUE" : "FALSE");
        m_summary_stream << separator << template_start;
        auto template_events =
                data.read_attrs.model_stride > 0
                        ? static_cast<uint64_t>(template_samples / data.read_attrs.model_stride)
                        : 0;
        m_summary_stream << separator << template_events;
        m_summary_stream << separator << template_duration;
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
        m_summary_stream << separator << get_tag<std::string>(record, "po", "not_set");
        m_summary_stream << separator
                         << (data.read_attrs.experiment_id.empty() ? "unknown"
                                                                   : data.read_attrs.experiment_id);
        m_summary_stream << separator << data.read_attrs.sample_id;
        m_summary_stream << separator << get_tag<std::string>(record, "er", "unknown");
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
            barcode_arrangement =
                    barcode_kits::normalize_barcode_name(data.barcoding_result->barcode_name);
            alias = data.barcoding_result->alias.empty() ? barcode_arrangement
                                                         : data.barcoding_result->alias;
            type = data.barcoding_result->type;
            barcode_kit = data.barcoding_result->kit;
            barcode_variant = get_tag<std::string>(data.bam_ptr.get(), "bv", "n/a");

            auto barcode_info = get_array<float>(record, "bi");
            if (barcode_info.size() == 7) {
                barcode_score = barcode_info[0];
                barcode_front_begin_index = static_cast<int>(barcode_info[1]);
                barcode_front_foundseq_length = static_cast<int>(barcode_info[2]);
                barcode_front_score = barcode_info[3];
                barcode_rear_end_index = static_cast<int>(barcode_info[4]);
                barcode_rear_foundseq_length = static_cast<int>(barcode_info[5]);
                barcode_rear_score = barcode_info[6];
            }
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
            if (m_dynamic_header != nullptr) {
                const auto& it = m_dynamic_header->find(data.read_attrs);
                if (it == m_dynamic_header->cend()) {
                    spdlog::error("Failed to find dynamic header: RG='{}', runid='{}'",
                                  utils::get_read_group_tag(data.bam_ptr.get()),
                                  data.read_attrs.protocol_run_id);
                    throw std::runtime_error("SummaryFileWriter - Failed to load dynamic header.");
                }
                alignment_genome =
                        sam_hdr_tid2name(it->second->get_merged_header(), record->core.tid);
            } else if (m_shared_header != nullptr) {
                alignment_genome = sam_hdr_tid2name(m_shared_header.get(), record->core.tid);
            }
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
            alignment_num_alignments = data.read_attrs.num_alignments;
            alignment_num_secondary_alignments = data.read_attrs.num_secondary_alignments;
            alignment_num_supplementary_alignments = data.read_attrs.num_supplementary_alignments;
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

SummaryFileWriter::ReadInitialiser::ReadInitialiser(sam_hdr_t* hdr, AlignmentCounts aln_counts)
        : m_header(hdr),
          m_alignment_counts(std::move(aln_counts)),
          m_read_groups(utils::parse_read_groups(m_header)),
          m_minimum_qscore(get_min_qscore(m_header)) {}

void SummaryFileWriter::ReadInitialiser::update_read_attributes(HtsData& data) const {
    if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
        const std::string rg_tag_value = bam_aux2Z(rg_tag);
        const auto& read_group = m_read_groups.at(rg_tag_value);
        data.read_attrs.protocol_run_id = read_group.run_id;
        data.read_attrs.flowcell_id = read_group.flowcell_id;
        data.read_attrs.experiment_id = read_group.experiment_id;
        data.read_attrs.sample_id = read_group.sample_id;
        data.read_attrs.position_id = read_group.position_id;
        data.read_attrs.model_stride = read_group.model_stride;

        if (const auto qs_tag = bam_aux_get(data.bam_ptr.get(), "qs"); qs_tag != nullptr) {
            const float qscore = static_cast<float>(bam_aux2f(qs_tag));
            data.read_attrs.is_status_pass = qscore >= m_minimum_qscore;
        }

        try {
            if (const auto st_tag = bam_aux_get(data.bam_ptr.get(), "st"); st_tag != nullptr) {
                const std::string read_start_time_str = bam_aux2Z(st_tag);
                const auto acq_start_time =
                        utils::get_unix_time_ms_from_string_timestamp(read_group.acq_start_time);
                const auto read_start_time =
                        utils::get_unix_time_ms_from_string_timestamp(read_start_time_str);
                data.read_attrs.start_time_ms = read_start_time - acq_start_time;
            }
        } catch (...) {
            // can't parse something, ignore start_time and continue
        }
    }
}

void SummaryFileWriter::ReadInitialiser::update_barcoding_fields(HtsData& data) const {
    if (const auto rg_tag = bam_aux_get(data.bam_ptr.get(), "RG"); rg_tag != nullptr) {
        const std::string rg_tag_value = bam_aux2Z(rg_tag);
        KString ks_wrapper(100000);
        auto& ks = ks_wrapper.get();

        data.barcoding_result = std::make_shared<BarcodeScoreResult>();
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "SM", &ks) == 0) {
            data.barcoding_result->barcode_name = std::string(ks.s, ks.l);
        }
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "al", &ks) == 0) {
            data.barcoding_result->alias = std::string(ks.s, ks.l);
        }
        if (sam_hdr_find_tag_id(m_header, "RG", "ID", rg_tag_value.c_str(), "bk", &ks) == 0) {
            data.barcoding_result->kit = std::string(ks.s, ks.l);
        }
    }
}

void SummaryFileWriter::ReadInitialiser::update_alignment_fields(HtsData& data) const {
    const auto alignment_counts_it = m_alignment_counts.find(bam_get_qname(data.bam_ptr.get()));
    if (alignment_counts_it != std::end(m_alignment_counts)) {
        const auto& counts = alignment_counts_it->second;
        data.read_attrs.num_alignments = counts[0];
        data.read_attrs.num_secondary_alignments = counts[1];
        data.read_attrs.num_supplementary_alignments = counts[2];
    }
}

void update_alignment_counts(const std::string& path,
                             SummaryFileWriter::AlignmentCounts& alignment_counts) {
    const auto file = dorado::HtsFilePtr(hts_open(path.c_str(), "r"));
    if (file->format.format != htsExactFormat::sam && file->format.format != htsExactFormat::bam) {
        return;
    }

    dorado::SamHdrPtr header(sam_hdr_read(file.get()));
    if (header->n_targets == 0) {
        return;
    }

    BamPtr record(bam_init1());
    while (sam_read1(file.get(), header.get(), record.get()) >= 0) {
        if (record->core.flag & BAM_FUNMAP) {
            continue;
        }
        auto& read_counts = alignment_counts[bam_get_qname(record.get())];
        if (record->core.flag & BAM_FSUPPLEMENTARY) {
            ++read_counts[2];
        }
        if (record->core.flag & BAM_FSECONDARY) {
            ++read_counts[1];
        }
        ++read_counts[0];
    }
}

}  // namespace dorado::hts_writer
