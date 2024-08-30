#include "HtsReader.h"

#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace dorado {

namespace {

class HtsLibBamRecordGenerator : public details::BamRecordGenerator {
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

    bool is_valid() { return m_file != nullptr && m_header != nullptr; }

    sam_hdr_t* header() const { return m_header.get(); }
    const std::string& format() const { return m_format; }

    bool try_get_next_record(bam1_t* record) override {
        return sam_read1(m_file.get(), m_header.get(), record) >= 0;
    }
};

class BamRecordGeneratorImpl : public details::BamRecordGenerator {
public:
    BamRecordGeneratorImpl(const std::string& filename) : m_hts_reader(filename) {}

    sam_hdr_t* header() const { return m_hts_reader.header(); }

    const std::string& format() const { return m_hts_reader.format(); }

    bool try_get_next_record(bam1_t* record) override {
        return m_hts_reader.try_get_next_record(record);
    }

    bool is_valid() { return m_hts_reader.is_valid(); }

private:
    HtsLibBamRecordGenerator m_hts_reader;
};

}  // namespace

HtsReader::HtsReader(const std::string& filename,
                     std::optional<std::unordered_set<std::string>> read_list)
        : m_client_info(std::make_shared<DefaultClientInfo>()), m_read_list(std::move(read_list)) {
    auto bam_record_generator = std::make_unique<BamRecordGeneratorImpl>(filename);
    if (!bam_record_generator->is_valid()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    m_header = bam_record_generator->header();
    m_format = bam_record_generator->format();
    is_aligned = m_header->n_targets > 0;
    m_bam_record_generator = std::move(bam_record_generator);

    record.reset(bam_init1());
}

HtsReader::~HtsReader() { record.reset(); }

void HtsReader::set_client_info(std::shared_ptr<ClientInfo> client_info) {
    m_client_info = std::move(client_info);
}

void HtsReader::set_record_mutator(std::function<void(BamPtr&)> mutator) {
    m_record_mutator = std::move(mutator);
}

bool HtsReader::read() { return m_bam_record_generator->try_get_next_record(record.get()); }

bool HtsReader::has_tag(const char* tagname) {
    uint8_t* tag = bam_aux_get(record.get(), tagname);
    return static_cast<bool>(tag);
}

std::size_t HtsReader::read(Pipeline& pipeline, std::size_t max_reads) {
    std::size_t num_reads = 0;
    while (this->read()) {
        if (m_read_list) {
            std::string read_id = bam_get_qname(record.get());
            if (m_read_list->find(read_id) == m_read_list->end()) {
                continue;
            }
        }
        if (m_record_mutator) {
            m_record_mutator(record);
        }

        BamMessage bam_message{BamPtr(bam_dup1(record.get())), m_client_info};
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

sam_hdr_t* HtsReader::header() const { return m_header; }

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
