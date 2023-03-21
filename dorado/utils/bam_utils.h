#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <set>
#include <string>
namespace dorado::utils {

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
std::map<std::string, std::shared_ptr<Read>> read_bam(const std::string& filename,
                                                      const std::set<std::string>& read_ids);

class BamReader {
public:
    BamReader(const std::string& filename);
    ~BamReader();
    bool next();
    char* m_format;
    bool m_is_aligned;
    bam1_t* m_record;
    sam_hdr_t* m_header;

private:
    htsFile* m_file;
};

class BamWriter {
public:
    BamWriter(const std::string& filename, const sam_hdr_t* header);
    ~BamWriter();
    int write_record(bam1_t* record);
    sam_hdr_t* m_header;

private:
    htsFile* m_file;
};

}  // namespace dorado::utils
