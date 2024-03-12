#include "BedFile.h"

#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <sstream>

namespace dorado::alignment {

BedFile::Entries const BedFile::NO_ENTRIES{};

const std::string & BedFile::filename() const { return m_file_name; }

bool BedFile::load(const std::string & bed_filename) {
    m_file_name = bed_filename;

    // Read in each line and parse it into the BedFile structure
    std::ifstream file_stream(m_file_name, std::ios::ate);
    if (file_stream.fail()) {
        spdlog::error("Failed to open Bed file for reading: '{}'", m_file_name);
        return false;
    }
    auto file_size = file_stream.tellg();
    file_stream.seekg(0);
    bool reading_header = true;
    bool allow_column_headers = true;
    while (!(file_stream.tellg() == (int32_t)file_size || file_stream.tellg() == -1)) {
        std::string bed_line;
        std::getline(file_stream, bed_line);

        if (utils::starts_with(bed_line, "#")) {
            continue;
        }
        // Remove whitespace from end of the line
        utils::rtrim(bed_line);
        // check for header markers and ignore if present
        if (reading_header) {
            if (utils::starts_with(bed_line, "browser") || utils::starts_with(bed_line, "track")) {
                continue;
            }
            reading_header = false;
        }

        // required fields from BED line
        std::string reference_name;
        size_t start, end;
        char strand = '.';
        std::istringstream bed_line_stream(bed_line);
        bed_line_stream >> reference_name >> start >> end;
        std::string next_col_str;
        // Guards against only the required fields being present
        if (!bed_line_stream.eof()) {
            // Strand is the sixth column so track column index
            int8_t c_index = 4;
            while (bed_line_stream >> next_col_str) {
                if (c_index < 6) {
                    c_index += 1;
                } else if (next_col_str == "+") {
                    strand = '+';
                    break;
                } else if (next_col_str == "-") {
                    strand = '-';
                    break;
                    // 6th column and not "+/-/."
                } else if (next_col_str != ".") {
                    spdlog::error("Invalid data reading strand from bed file '{}'", m_file_name);
                    return false;
                }
                // No strand column present and we're at the end
                if (bed_line_stream.eof()) {
                    break;
                }
            }
        }
        if (bed_line_stream.fail() && !bed_line_stream.eof()) {
            if (allow_column_headers) {
                allow_column_headers = false;
                spdlog::info(
                        "Invalid data found reading bed file '{}'. Assuming column headers and "
                        "skipping line.",
                        m_file_name);
                continue;
            }
            spdlog::error("Invalid data reading bed file '{}'", m_file_name);
            return false;
        }
        allow_column_headers = false;
        m_genomes[reference_name].push_back({bed_line, start, end, strand});
    }

    return true;
};

const BedFile::Entries & BedFile::entries(const std::string & genome) const {
    auto it = m_genomes.find(genome);
    return it != m_genomes.end() ? it->second : NO_ENTRIES;
}

}  // namespace dorado::alignment
