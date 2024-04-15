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

HtsReader::HtsReader(const std::string& filename,
                     std::optional<std::unordered_set<std::string>> read_list)
        : m_read_list(std::move(read_list)), m_client_info(std::make_shared<DefaultClientInfo>()) {
    m_file = hts_open(filename.c_str(), "r");
    if (!m_file) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    // If input format is FASTX, read tags from the query name line.
    hts_set_opt(m_file, FASTQ_OPT_AUX, "1");
    format = hts_format_description(hts_get_format(m_file));
    header = sam_hdr_read(m_file);
    if (!header) {
        throw std::runtime_error("Could not read header from file: " + filename);
    }
    is_aligned = header->n_targets > 0;
    record.reset(bam_init1());
}

HtsReader::~HtsReader() {
    hts_free(format);
    sam_hdr_destroy(header);
    record.reset();
    hts_close(m_file);
}

void HtsReader::set_client_info(std::shared_ptr<ClientInfo> client_info) {
    m_client_info = std::move(client_info);
}

bool HtsReader::read() { return sam_read1(m_file, header, record.get()) >= 0; }

bool HtsReader::has_tag(std::string tagname) {
    uint8_t* tag = bam_aux_get(record.get(), tagname.c_str());
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
