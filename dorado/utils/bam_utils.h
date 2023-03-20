#pragma once
#include "htslib/sam.h"
#include "read_pipeline/ReadPipeline.h"

#include <map>
#include <string>
namespace dorado::utils {

/**
 * @brief Reads a BAM file and returns a map of read IDs to Read objects.
 *
 * This function opens a BAM file specified by the input filename parameter,
 * reads the alignments, and creates a map that associates read IDs with their
 * corresponding Read objects. The Read objects contain the read ID, sequence,
 * and quality string.
 *
 * @param filename The input BAM file path as a string.
 * @return A map with read IDs as keys and shared pointers to Read objects as values.
 *
 * @note The caller is responsible for managing the memory of the returned map.
 * @note The input BAM file must be properly formatted and readable.
 */
std::map<std::string, std::shared_ptr<Read>> read_bam(const std::string& filename);

class BamReader {
public:
    BamReader(const std::string& filename);
    ~BamReader();
    bool next();
    bam1_t* m_record;
    sam_hdr_t* m_header;

private:
    htsFile* m_file;
    hts_itr_t* m_iterator;
};

}  // namespace dorado::utils
