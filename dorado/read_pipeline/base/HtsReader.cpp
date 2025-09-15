#include "read_pipeline/base/HtsReader.h"

#include "hts_utils/HeaderMapper.h"
#include "hts_utils/bam_utils.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/base/ReadPipeline.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace dorado {

namespace {

class HtsLibBamRecordGenerator {
    HtsFilePtr m_file{};
    SamHdrPtr m_header{};
    std::string m_format{};

public:
    HtsLibBamRecordGenerator(const std::string& filename) {
        m_file.reset(hts_open(filename.c_str(), "r"));
        if (!m_file) {
            return;
        }
        // If input format is FASTX, read tags from the query name line.
        hts_set_opt(m_file.get(), FASTQ_OPT_AUX, "1");
        auto format = hts_format_description(hts_get_format(m_file.get()));
        if (format) {
            m_format = format;
            hts_free(format);
        }
        m_header.reset(sam_hdr_read(m_file.get()));
        if (!m_header) {
            return;
        }
    }

    bool is_valid() const { return m_file != nullptr && m_header != nullptr; }

    sam_hdr_t* header() const { return m_header.get(); }
    const std::string& format() const { return m_format; }

    bool try_get_next_record(bam1_t& record) {
        return sam_read1(m_file.get(), m_header.get(), &record) >= 0;
    }
};

// This function allows us to map the reference id from input BAM records to what
// they should be in the output file, based on the new ordering of references in
// the merged header.
void adjust_tid(const std::vector<uint32_t>& mapping, BamPtr& record) {
    auto tid = record.get()->core.tid;
    if (tid >= 0) {
        if (tid >= int32_t(mapping.size())) {
            throw std::range_error("BAM tid field out of range with respect to SQ lines.");
        }
        record.get()->core.tid = int32_t(mapping.at(tid));
    }
}

}  // namespace

HtsReader::HtsReader(const std::string& filename,
                     std::optional<std::unordered_set<std::string>> read_list)
        : m_filename(filename),
          m_client_info(std::make_shared<DefaultClientInfo>()),
          m_read_list(std::move(read_list)) {
    if (!try_initialise_generator<HtsLibBamRecordGenerator>(m_filename)) {
        throw std::runtime_error("Could not open file: " + m_filename);
    }
    is_aligned = m_header->n_targets > 0;

    record.reset(bam_init1());
}

template <typename T>
bool HtsReader::try_initialise_generator(const std::string& filepath) {
    auto generator = std::make_shared<T>(filepath);  // shared to allow copy assignment
    if (!generator->is_valid()) {
        return false;
    }
    m_header = generator->header();
    m_format = generator->format();
    m_bam_record_generator = [generator_ = std::move(generator),
                              filename = std::filesystem::path(filepath).filename().string(),
                              this](bam1_t& bam_record) {
        if (!generator_->try_get_next_record(bam_record)) {
            return false;
        }

        // If the record doesn't have a filename set then say that it came from the currently processing file.
        if (m_add_filename_tag && !bam_aux_get(&bam_record, "fn")) {
            bam_aux_append(&bam_record, "fn", 'Z', static_cast<int>(filename.size() + 1),
                           reinterpret_cast<const uint8_t*>(filename.c_str()));
        }

        return true;
    };
    return true;
}

void HtsReader::set_client_info(std::shared_ptr<ClientInfo> client_info) {
    m_client_info = std::move(client_info);
}

bool HtsReader::read() { return m_bam_record_generator(*record); }

bool HtsReader::has_tag(const char* tagname) {
    uint8_t* tag = bam_aux_get(record.get(), tagname);
    return static_cast<bool>(tag);
}

std::size_t HtsReader::read(Pipeline& pipeline,
                            std::size_t max_reads,
                            const bool strip_alignments,
                            const std::unique_ptr<utils::HeaderMapper> header_mapper,
                            const bool skip_sec_supp) {
    std::size_t num_reads = 0;
    while (this->read()) {
        if (m_read_list) {
            std::string read_id = bam_get_qname(record.get());
            if (m_read_list->find(read_id) == m_read_list->end()) {
                continue;
            }
        }

        if (skip_sec_supp &&
            ((record->core.flag & BAM_FSECONDARY) || (record->core.flag & BAM_FSUPPLEMENTARY))) {
            continue;
        }

        std::unique_ptr<HtsData> hts_data;
        if (header_mapper == nullptr) {
            hts_data = std::make_unique<HtsData>(HtsData{BamPtr(bam_dup1(record.get()))});
        } else {
            // Get read attributes by read group ID
            const auto& read_attrs = header_mapper->get_read_attributes(record.get());

            if (!strip_alignments) {
                const auto& sq_mapping =
                        header_mapper->get_merged_header(read_attrs).get_sq_mapping(m_filename);
                adjust_tid(sq_mapping, record);
            }

            hts_data =
                    std::make_unique<HtsData>(HtsData{BamPtr(bam_dup1(record.get())), read_attrs});
        }

        BamMessage bam_message{std::move(hts_data), m_client_info};
        pipeline.push_message(std::move(bam_message));

        ++num_reads;
        if (max_reads > 0 && num_reads >= max_reads) {
            break;
        }
        if (num_reads % 50000 == 0) {
            spdlog::debug("Processed {} reads", num_reads);
        }
    }
    spdlog::debug("Total reads processed: {}", num_reads);
    return num_reads;
}

sam_hdr_t* HtsReader::header() { return m_header; }

const std::string& HtsReader::format() const { return m_format; }

ReadMap read_bam(const std::string& filename, const std::unordered_set<std::string>& read_ids) {
    HtsReader reader(filename, std::nullopt);

    ReadMap reads;

    while (reader.read()) {
        std::string read_id = bam_get_qname(reader.record);

        if (read_ids.find(read_id) == read_ids.end()) {
            continue;
        }

        uint8_t* qstring = bam_get_qual(reader.record);
        uint8_t* sequence = bam_get_seq(reader.record);

        uint32_t seqlen = reader.record->core.l_qseq;
        std::vector<uint8_t> qualities(seqlen);
        std::vector<char> nucleotides(seqlen);

        // Todo - there is a better way to do this.
        for (uint32_t i = 0; i < seqlen; i++) {
            qualities[i] = qstring[i] + 33;
            nucleotides[i] = seq_nt16_str[bam_seqi(sequence, i)];
        }

        auto tmp_read = std::make_unique<SimplexRead>();
        tmp_read->read_common.read_id = read_id;
        tmp_read->read_common.seq = std::string(nucleotides.begin(), nucleotides.end());
        tmp_read->read_common.qstring = std::string(qualities.begin(), qualities.end());
        reads[read_id] = std::move(tmp_read);
    }

    return reads;
}

std::unordered_set<std::string> fetch_read_ids(const std::string& filename) {
    if (filename.empty()) {
        return {};
    }
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error("Resume file cannot be found: " + filename);
    }

    auto initial_hts_log_level = hts_get_log_level();
    hts_set_log_level(HTS_LOG_OFF);

    std::unordered_set<std::string> read_ids;
    HtsReader reader(filename, std::nullopt);
    try {
        while (reader.read()) {
            std::string read_id = bam_get_qname(reader.record);
            read_ids.insert(read_id);
        }
    } catch (std::exception&) {
        // Do nothing.
    }

    hts_set_log_level(initial_hts_log_level);

    return read_ids;
}

}  // namespace dorado
