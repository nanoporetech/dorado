#include "bed_file.h"

#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <tuple>

namespace dorado::alignment {
namespace {
constexpr std::size_t MIN_COLS{3};
constexpr std::size_t MAX_COLS{12};

std::vector<std::string> get_tokens(const std::string& bed_line) {
    std::vector<std::string> result{};
    std::istringstream bed_line_stream(bed_line);

    // if too many columns include an extra column so caller can validate and find the error
    for (std::size_t col{0}; col <= MAX_COLS; ++col) {
        std::string token;
        if (!std::getline(bed_line_stream, token, '\t')) {
            break;
        }
        result.push_back(std::move(token));
    }

    return result;
}

template <typename T>
bool try_get(const std::string& token, T& target) {
    std::istringstream token_stream(token);
    T result;
    if (!(token_stream >> result)) {
        return false;
    }
    std::swap(target, result);
    return true;
}

std::optional<BedFile::Entry> get_entry_from_bedline(const std::vector<std::string>& tokens) {
    if (tokens.size() < MIN_COLS || tokens.size() > MAX_COLS) {
        return std::nullopt;
    }
    BedFile::Entry result;
    if (!try_get(tokens[1], result.start)) {
        return std::nullopt;
    }
    if (!try_get(tokens[2], result.end)) {
        return std::nullopt;
    }
    if (tokens.size() >= 6) {
        if (!try_get(tokens[5], result.strand)) {
            return std::nullopt;
        }
    } else {
        result.strand = '.';
    }

    return result;
}

}  // namespace

bool operator==(const BedFile::Entry& l, const BedFile::Entry& r) {
    return std::tie(l.bed_line, l.start, l.end, l.strand) ==
           std::tie(r.bed_line, r.start, r.end, r.strand);
}

bool operator!=(const BedFile::Entry& l, const BedFile::Entry& r) { return !(l == r); }

const BedFile::Entries BedFile::NO_ENTRIES{};

const std::string& BedFile::filename() const { return m_file_name; }

bool BedFile::load(const std::string& bed_filename) {
    m_file_name = bed_filename;

    // Read in each line and parse it into the BedFile structure
    std::ifstream file_stream(m_file_name);
    if (file_stream.fail()) {
        spdlog::error("Failed to open Bed file for reading: '{}'", m_file_name);
        return false;
    }
    return load(file_stream);
};

bool BedFile::load(std::istream& input_stream) {
    std::string bed_line;

    bool reading_header = true;
    bool allow_column_headers = true;
    while (std::getline(input_stream, bed_line)) {
        if (bed_line.empty() || utils::starts_with(bed_line, "#")) {
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
        auto tokens = get_tokens(bed_line);
        auto entry = get_entry_from_bedline(tokens);
        if (!entry) {
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
        entry->bed_line = std::move(bed_line);
        allow_column_headers = false;
        m_genomes[tokens[0]].push_back(std::move(*entry));

        //// required fields from BED line
        //std::string reference_name;
        //size_t start, end;
        //char strand = '.';
        //std::istringstream bed_line_stream(bed_line);
        //bed_line_stream >> reference_name >> start >> end;
        //std::string next_col_str;
        //// Guards against only the required fields being present
        //if (!bed_line_stream.eof()) {
        //    // Strand is the sixth column so track column index
        //    int8_t c_index = 4;
        //    while (bed_line_stream >> next_col_str) {
        //        if (c_index < 6) {
        //            c_index += 1;
        //        } else if (next_col_str == "+") {
        //            strand = '+';
        //            break;
        //        } else if (next_col_str == "-") {
        //            strand = '-';
        //            break;
        //            // 6th column and not "+/-/."
        //        } else if (next_col_str != ".") {
        //            spdlog::error("Invalid data reading strand from bed file '{}'", m_file_name);
        //            return false;
        //        }
        //        // No strand column present and we're at the end
        //        if (bed_line_stream.eof()) {
        //            break;
        //        }
        //    }
        //}
        //if (bed_line_stream.fail() && !bed_line_stream.eof()) {
        //    if (allow_column_headers) {
        //        allow_column_headers = false;
        //        spdlog::info(
        //                "Invalid data found reading bed file '{}'. Assuming column headers and "
        //                "skipping line.",
        //                m_file_name);
        //        continue;
        //    }
        //    spdlog::error("Invalid data reading bed file '{}'", m_file_name);
        //    return false;
        //}
    }

    return true;
}

//bool load2(std::istream& input_stream) {
//    input_stream.seekg(0, std::ios::end);
//    auto file_size = input_stream.tellg();
//    input_stream.seekg(0);
//    bool reading_header = true;
//    bool allow_column_headers = true;
//    while (!(input_stream.tellg() == (int32_t)file_size || input_stream.tellg() == -1)) {
//        std::string bed_line;
//        std::getline(input_stream, bed_line);
//
//        if (utils::starts_with(bed_line, "#")) {
//            continue;
//        }
//        // Remove whitespace from end of the line
//        utils::rtrim(bed_line);
//        // check for header markers and ignore if present
//        if (reading_header) {
//            if (utils::starts_with(bed_line, "browser") || utils::starts_with(bed_line, "track")) {
//                continue;
//            }
//            reading_header = false;
//        }
//
//        // required fields from BED line
//        std::string reference_name;
//        size_t start, end;
//        char strand = '.';
//        std::istringstream bed_line_stream(bed_line);
//        bed_line_stream >> reference_name >> start >> end;
//        std::string next_col_str;
//        // Guards against only the required fields being present
//        if (!bed_line_stream.eof()) {
//            // Strand is the sixth column so track column index
//            int8_t c_index = 4;
//            while (bed_line_stream >> next_col_str) {
//                if (c_index < 6) {
//                    c_index += 1;
//                } else if (next_col_str == "+") {
//                    strand = '+';
//                    break;
//                } else if (next_col_str == "-") {
//                    strand = '-';
//                    break;
//                    // 6th column and not "+/-/."
//                } else if (next_col_str != ".") {
//                    spdlog::error("Invalid data reading strand from bed file '{}'", m_file_name);
//                    return false;
//                }
//                // No strand column present and we're at the end
//                if (bed_line_stream.eof()) {
//                    break;
//                }
//            }
//        }
//        if (bed_line_stream.fail() && !bed_line_stream.eof()) {
//            if (allow_column_headers) {
//                allow_column_headers = false;
//                spdlog::info(
//                        "Invalid data found reading bed file '{}'. Assuming column headers and "
//                        "skipping line.",
//                        m_file_name);
//                continue;
//            }
//            spdlog::error("Invalid data reading bed file '{}'", m_file_name);
//            return false;
//        }
//        allow_column_headers = false;
//        m_genomes[reference_name].push_back({bed_line, start, end, strand});
//    }
//
//    return true;
//}

const BedFile::Entries& BedFile::entries(const std::string& genome) const {
    auto it = m_genomes.find(genome);
    return it != m_genomes.end() ? it->second : NO_ENTRIES;
}

}  // namespace dorado::alignment
