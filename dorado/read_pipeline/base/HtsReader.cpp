#include "read_pipeline/base/HtsReader.h"

#include "hts_utils/HeaderMapper.h"
#include "hts_utils/bam_utils.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/base/messages/SimplexRead.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace dorado {

namespace {

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
                     std::optional<std::unordered_set<std::string>> read_list,
                     std::size_t num_threads)
        : m_filename(filename),
          m_current_filename(std::filesystem::path(m_filename).filename().string()),
          m_client_info(std::make_shared<DefaultClientInfo>()),
          m_read_list(std::move(read_list)) {
    if (!open_file(num_threads)) {
        throw std::runtime_error("Could not open file: " + m_filename);
    }
    record.reset(bam_init1());
}

HtsReader::~HtsReader() = default;

void HtsReader::set_client_info(std::shared_ptr<ClientInfo> client_info) {
    m_client_info = std::move(client_info);
}

bool HtsReader::read() {
    if (sam_read1(m_file.get(), m_header.get(), record.get()) < 0) {
        return false;
    }

    // If the record doesn't have a filename set then say that it came from the currently processing file.
    if (m_add_filename_tag && !bam_aux_get(record.get(), "fn")) {
        bam_aux_append(record.get(), "fn", 'Z', static_cast<int>(m_current_filename.size() + 1),
                       reinterpret_cast<const uint8_t*>(m_current_filename.c_str()));
    }

    return true;
}

bool HtsReader::has_tag(const char* tagname) {
    uint8_t* tag = bam_aux_get(record.get(), tagname);
    return static_cast<bool>(tag);
}

std::size_t HtsReader::read(Pipeline& pipeline,
                            std::size_t max_reads,
                            const bool strip_alignments,
                            const utils::HeaderMapper* header_mapper,
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
            const auto input_read_group_id = utils::get_read_group_tag(record.get());
            const auto output_read_group_id =
                    header_mapper->get_output_read_group_id(m_filename, input_read_group_id);
            if (output_read_group_id != input_read_group_id) {
                bam_aux_update_str(record.get(), "RG",
                                   static_cast<int>(output_read_group_id.length() + 1),
                                   output_read_group_id.c_str());
            }

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
        for (const auto& initialiser : m_read_initialisers) {
            initialiser(*bam_message.data);
        }

        pipeline.push_message(std::move(bam_message));

        ++num_reads;
        if (max_reads > 0 && num_reads >= max_reads) {
            break;
        }
        if (num_reads % 50000 == 0) {
            DORADO_LOG(debug, "Processed {} reads", num_reads);
        }
    }
    DORADO_LOG(debug, "Total reads processed: {}", num_reads);
    return num_reads;
}

htsExactFormat HtsReader::exact_format() const { return hts_get_format(m_file.get())->format; }

std::string HtsReader::format_str() const {
    std::string format_str;
    auto format = hts_format_description(hts_get_format(m_file.get()));
    if (format) {
        format_str = format;
        hts_free(format);
    }
    return format_str;
}

bool HtsReader::open_file(std::size_t num_threads) {
    m_file.reset(hts_open(m_filename.c_str(), "r"));
    if (!m_file) {
        return false;
    }

    // If input format is FASTX, read tags from the query name line.
    hts_set_opt(m_file.get(), FASTQ_OPT_AUX, "1");

    // Read the header before enabling threading otherwise we can fail
    // to load it if the file is corrupt. See DOR-1634.
    m_header.reset(sam_hdr_read(m_file.get()));
    if (!m_header) {
        return false;
    }

    // Enable multithreaded loading if asked to do so.
    if (num_threads > 1) {
        hts_set_threads(m_file.get(), num_threads);
    }

    return true;
}

ReadMap read_bam(const std::string& filename,
                 const std::unordered_set<std::string>& read_ids,
                 std::size_t num_threads) {
    HtsReader reader(filename, std::nullopt, num_threads);

    ReadMap reads;

    while (reader.read()) {
        std::string read_id = bam_get_qname(reader.record);

        if (read_ids.find(read_id) == read_ids.end()) {
            continue;
        }

        uint8_t* qstring = bam_get_qual(reader.record);
        uint8_t* sequence = bam_get_seq(reader.record);

        uint32_t seqlen = reader.record->core.l_qseq;
        std::string qualities(seqlen, '\0');
        std::string nucleotides(seqlen, '\0');

        // Todo - there is a better way to do this.
        for (uint32_t i = 0; i < seqlen; i++) {
            qualities[i] = static_cast<char>(qstring[i] + 33);
            nucleotides[i] = seq_nt16_str[bam_seqi(sequence, i)];
        }

        auto tmp_read = std::make_unique<SimplexRead>();
        tmp_read->read_common.read_id = read_id;
        tmp_read->read_common.seq = std::move(nucleotides);
        tmp_read->read_common.qstring = std::move(qualities);
        reads[std::move(read_id)] = std::move(tmp_read);
    }

    return reads;
}

}  // namespace dorado
