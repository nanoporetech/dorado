#include "ResumeLoaderNode.h"

#include "HtsReader.h"

#include <htslib/sam.h>
#include <indicators/indeterminate_progress_bar.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <memory>

namespace dorado {

ResumeLoaderNode::ResumeLoaderNode(MessageSink& sink, const std::string& resume_file)
        : m_sink(sink), m_resume_file(resume_file) {
    if (!std::filesystem::exists(resume_file)) {
        throw std::runtime_error("Resume file cannot be found: " + resume_file);
    }
}

void ResumeLoaderNode::copy_completed_reads() {
    indicators::IndeterminateProgressBar bar{indicators::option::BarWidth{20},
                                             indicators::option::Start{"["},
                                             indicators::option::Fill{"·"},
                                             indicators::option::Lead{"<=>"},
                                             indicators::option::End{"]"},
                                             indicators::option::PostfixText{"Resuming from file"},
                                             indicators::option::Stream{std::cerr}};

    // Turn off logging for warnings.
    auto initial_hts_log_level = hts_get_log_level();
    hts_set_log_level(HTS_LOG_OFF);

    auto reader = std::make_unique<HtsReader>(m_resume_file);

    // Iterate over all reads and write to sink.
    try {
        while (reader->read()) {
            std::string read_id = bam_get_qname(reader->record);
            m_processed_read_ids.insert(read_id);
            m_sink.push_message(BamPtr(bam_dup1(reader->record.get())));
            if (m_processed_read_ids.size() % 100 == 0) {
                bar.tick();
            }
        }
    } catch (std::exception& e) {
        // Exception implies the reader could not read
        // the last record. We take this to be the end of
        // properly formatted records.
    }
    std::cerr << "\r";
    spdlog::info("> {} reads found in resume file.", m_processed_read_ids.size());

    hts_set_log_level(initial_hts_log_level);
}

std::unordered_set<std::string> ResumeLoaderNode::get_processed_read_ids() const {
    return m_processed_read_ids;
}

}  // namespace dorado
