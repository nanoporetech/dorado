#pragma once

#include "utils/types.h"

#include <bitset>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dorado::utils {

class SampleSheet {
public:
    enum class Type { none, barcode };
    enum IndexBits {
        FLOW_CELL_ID = 0,
        POSITION_ID,
    };

    SampleSheet();

    // Calls load on the passed filename, if it is not empty.
    // If skip_index_matching is true the lookup by flowcell/experiment id will be skipped when fetching an alias.
    // In this case, the constructor will throw if the sample sheet contains entries for more that one flow_cell_id,
    // position_id or experiment_id, or if any barcode is re-used.
    explicit SampleSheet(const std::string& filename, bool skip_index_matching);

    // load a sample sheet from a file. Throws a std::runtime_error for any failure condition.
    void load(const std::string& filename);

    // (Testability) load a sample sheet from the given stream.
    // Throws a std::runtime_error for any failure condition using the filename in the
    // error message.
    void load(std::istream& input_stream, const std::string& filename);

    // Return the sample sheet filename.
    const std::string& get_filename() const { return m_filename; }

    // Return the sample sheet type based on the column headers it contains.
    Type get_type() const { return m_type; }

    bool contains_column(const std::string& column) const { return m_col_indices.count(column); }

    // For a given flow_cell_id, position_id, experiment_id and barcode, get the named alias.
    //  Returns an empty string if one does not exist in the loaded sample sheet, or if the sample
    //  sheet is not of type "barcode".
    std::string get_alias(const std::string& flow_cell_id,
                          const std::string& position_id,
                          const std::string& experiment_id,
                          const std::string& barcode) const;

    /**
     * Get all of the barcodes that are present in the sample sheet.
     * @return All of the barcodes that are present, or std::nullopt if the sample sheet is not loaded.
     */
    BarcodeFilterSet get_barcode_values() const;

    /**
     * Check whether the a list of allowed barcodes is set and, if so, whether the provided barcode is in it.
     */
    bool barcode_is_permitted(const std::string& barcode_name) const;

private:
    using Row = std::vector<std::string>;
    std::string m_filename;
    Type m_type{Type::none};
    std::bitset<2> m_index{};
    std::unordered_map<std::string, size_t> m_col_indices;
    std::vector<Row> m_rows;
    bool m_skip_index_matching;
    BarcodeFilterSet m_allowed_barcodes;

    void validate_headers(const std::vector<std::string>& col_names, const std::string& filename);
    bool check_index(const std::string& flow_cell_id, const std::string& position_id) const;
    bool match_index(const Row& row,
                     const std::string& flow_cell_id,
                     const std::string& position_id,
                     const std::string& experiment_id) const;
    std::string get(const Row& row, const std::string& key) const;
    void validate_text(const Row& row, const std::string& key) const;
    void validate_alias(const Row& row, const std::string& key) const;
    bool is_barcode_mapping_unique() const;
};

std::string to_string(SampleSheet::Type type);

namespace details {

enum class EolFileFormat {  // identifiers decorated with "_eol" suffix to avoid likely #defines
    linux_eol,              //  '\n'
    windows_eol,            //  '\r` + '\n'
    osx_eol,                //  '\r'    (older osx style)
};

/**
 * If no EOL returns linux_eol as a default.
 * @remarks will set the stream position to 0 on exit, this is sufficient behaviour for
 * sample sheet usage.
 * @remarks Note that the same file may be identified as different format on different platforms
 * for example a windows file will be identified by this check as windows when running on linux
 * but will be identified as linux when running on windows. This is because the file is not opened
 * as binary and so new line conversion operations will be applied, i.e. windows converts CR + NL to
 * the NL character, unless opened for binary reading. But for our purposes we are not interested in
 * the underlying truth of the file format just the style in which it appears in the stream.
 */
EolFileFormat get_eol_file_format(std::istream& input);

}  // namespace details

}  // namespace dorado::utils
