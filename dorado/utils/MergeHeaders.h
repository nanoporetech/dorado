#pragma once
#include "types.h"

#include <map>
#include <set>
#include <vector>

struct sam_hdr_t;

namespace dorado::utils {

class MergeHeaders {
public:
    /** Construct an object for merging headers.
     *  @param strip_alignment If set, no SQ lines will be included in the
     *         merged header, and no checks will be made for SQ conflicts. 
     */
    MergeHeaders(bool strip_alignment);

    /** Add a header.
     *  @param hdr The header to add.
     *  @param filename The name that should be reported if any errors
     *         occur while trying to merge in the header data.
     *  @return An error string indicating what went wrong. If the header
     *          was successfully merged this will be empty.
     * 
     *  The HD line from the first header added will be used
     *  as the HD line for the merged header.
     * 
     *  If any RG or SQ lines conflict with ones that have already been
     *  merged, this will result in an error.
     */
    std::string add_header(sam_hdr_t* hdr, const std::string& filename);

    // Call this when you have added all the headers.
    void finalize_merge();

    /** Get the merged header object.
     * 
     *  This should only be called after finalize_merge() has been called.
     *  An exception will be thrown if finalize_merge() hasn't been called.
     * 
     *  The returned pointer is owned by the MergeHeaders object. Copy
     *  it with sam_hdr_dup() if you need a copy that will remain valid
     *  beyond the lifetime of the MergeHeaders object.
     */
    sam_hdr_t* get_merged_header() const;

    /** Get the mapping of indexes of SQ lines from original headers to
     *  the merged header.
     *  
     *  This should only be called after finalize_merge() has been called.
     *  An exception will be thrown if finalize_merge() hasn't been called.
     * 
     *  Each element of the returned vector corresponds to one of the
     *  original header files, in the order they were added.
     * 
     *  Each element is itself a vector, containing the numerical SQ
     *  line ids for the merged header. So if, in the second input header
     *  the numeric SQ line id for a read was 7, then to get the numeric
     *  SQ line id for the merged header, you would use:
     *  
     *  auto mapping = merger.get_sq_mapping();
     *  auto n = mapping[1][7];
     * 
     *  This should be used to update records in the merged BAM file, so
     *  that their tid fields refer to the correct references in the
     *  merged file.
     */
    std::vector<std::vector<uint32_t>> get_sq_mapping() const;

private:
    struct RefInfo {
        uint32_t index;         // Index of SQ line in original header.
        std::string line_text;  // Full SQ line text.
    };

    bool m_strip_alignment;
    SamHdrPtr m_merged_header;
    std::vector<std::vector<uint32_t>> m_sq_mapping;

    // Stores unique RG lines across all headers.
    std::map<std::string, std::string> m_read_group_lut;

    // Stores unique PG lines across all headers.
    // Key = PG line id. Value = all unique lines seen with that id.
    std::map<std::string, std::set<std::string>> m_program_lut;

    // Stores unique SQ lines across all headers.
    // Key = ref name, Value = line text for SQ line.
    std::map<std::string, std::string> m_ref_lut;

    // Stores all unique non-SQ/PG/RG lines across all headers.
    std::set<std::string> m_other_lines;

    // Stores SQ line data for each header.
    // One entry for each header. Key = ref name, Value = info for ref.
    std::vector<std::map<std::string, RefInfo>> m_ref_info_lut;

    int check_and_add_ref_data(sam_hdr_t* hdr);
    int check_and_add_rg_data(sam_hdr_t* hdr);
    int add_pg_data(sam_hdr_t* hdr);
    void add_other_lines(sam_hdr_t* hdr);
};

}  // namespace dorado::utils
