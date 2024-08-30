#pragma once

#include "read_pipeline/ClientInfo.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <htslib/sam.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dorado {

using ReadMap = std::unordered_map<std::string, SimplexReadPtr>;

class Pipeline;

class HtsReader {
public:
    HtsReader(const std::string& filename,
              std::optional<std::unordered_set<std::string>> read_list);
    ~HtsReader();
    bool read();

    // If reading directly into a pipeline need to set the client info on the messages
    void set_client_info(std::shared_ptr<ClientInfo> client_info);
    std::size_t read(Pipeline& pipeline, std::size_t max_reads);
    template <typename T>
    T get_tag(const char* tagname);
    bool has_tag(const char* tagname);
    void set_record_mutator(std::function<void(BamPtr&)> mutator);

    char* format{nullptr};
    bool is_aligned{false};
    BamPtr record{nullptr};

    sam_hdr_t* header() const { return m_header; }

private:
    htsFile* m_file{nullptr};
    sam_hdr_t* m_header{nullptr};
    std::shared_ptr<ClientInfo> m_client_info;

    std::function<void(BamPtr&)> m_record_mutator{};
    std::optional<std::unordered_set<std::string>> m_read_list;
};

template <typename T>
T HtsReader::get_tag(const char* tagname) {
    T tag_value{};
    uint8_t* tag = bam_aux_get(record.get(), tagname);

    if (!tag) {
        return tag_value;
    }
    if constexpr (std::is_integral_v<T>) {
        tag_value = static_cast<T>(bam_aux2i(tag));
    } else if constexpr (std::is_floating_point_v<T>) {
        tag_value = static_cast<T>(bam_aux2f(tag));
    } else {
        tag_value = static_cast<T>(bam_aux2Z(tag));
    }

    return tag_value;
}

/**
 * @brief Reads a SAM/BAM/CRAM file and returns a map of read IDs to Read objects.
 *
 * This function opens a SAM/BAM/CRAM file specified by the input filename parameter,
 * reads the alignments, and creates a map that associates read IDs with their
 * corresponding Read objects. The Read objects contain the read ID, sequence,
 * and quality string.
 *
 * @param filename The input BAM file path as a string.
 * @param read_ids A set of read_ids to filter on.
 * @return A map with read IDs as keys and shared pointers to Read objects as values.
 *
 * @note The caller is responsible for managing the memory of the returned map.
 * @note The input BAM file must be properly formatted and readable.
 */
ReadMap read_bam(const std::string& filename, const std::unordered_set<std::string>& read_ids);

/**
 * @brief Reads an HTS file format (SAM/BAM/FASTX/etc) and returns a set of read ids.
 *
 * This function opens the HTS file using the htslib APIs and iterates through
 * all records. When an unreadable record is encountered, the iteration is stopped
 * and all read ids seen so far are returned.
 *
 * @param filename The path to the input HTS file.
 * @return An unordered set with read ids.
 */
std::unordered_set<std::string> fetch_read_ids(const std::string& filename);

}  // namespace dorado
