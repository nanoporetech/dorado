#include "bed_file.h"

#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string_view>
#include <tuple>

namespace dorado::alignment {
namespace {
constexpr std::size_t MIN_COLS{3};
constexpr std::size_t MAX_COLS{12};

std::optional<std::vector<std::string>> get_tokens(const std::string& bed_line) {
    std::vector<std::string> result{};
    std::istringstream bed_line_stream(bed_line);

    // if too many columns include an extra column so caller can validate and find the error
    for (std::size_t col{0}; col <= MAX_COLS; ++col) {
        std::string token;
        if (!std::getline(bed_line_stream, token, '\t')) {
            break;
        }
        if (token.empty()) {
            return std::nullopt;
        }
        result.push_back(std::move(token));
    }

    if (result.empty()) {
        return std::nullopt;
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

bool try_get_strand(const std::string& token, char& target) {
    if (token.size() != 1) {
        return false;
    }
    const auto& candidate = token.at(0);
    if (candidate == '+' || candidate == '-' || candidate == '.') {
        target = candidate;
        return true;
    }
    return false;
}

bool try_get_entry_from_bedline(std::string bed_line, std::string& genome, BedFile::Entry& entry) {
    auto tokens = get_tokens(bed_line);
    if (!tokens || tokens->size() < MIN_COLS || tokens->size() > MAX_COLS) {
        return false;
    }

    BedFile::Entry result{};
    if (!try_get((*tokens)[1], result.start) || !try_get((*tokens)[2], result.end)) {
        return false;
    }
    if (tokens->size() >= 6 && !try_get_strand((*tokens)[5], result.strand)) {
        return false;
    }

    result.bed_line = std::move(bed_line);
    std::swap(entry, result);
    std::swap(genome, (*tokens)[0]);

    return true;
}

bool is_header_line(const std::string& candidate) {
    return utils::starts_with(candidate, "#") || utils::starts_with(candidate, "browser") ||
           utils::starts_with(candidate, "track");
}

}  // namespace

bool operator==(const BedFile::Entry& l, const BedFile::Entry& r) {
    auto l_line_view = utils::rtrim_view(l.bed_line);
    auto r_line_view = utils::rtrim_view(r.bed_line);
    return std::tie(l_line_view, l.start, l.end, l.strand) ==
           std::tie(r_line_view, r.start, r.end, r.strand);
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

    auto is_header = [header_section = true](const std::string& candidate) mutable {
        if (!header_section) {
            return false;
        }
        if (is_header_line(candidate)) {
            return true;
        }
        header_section = false;
        return false;
    };

    while (std::getline(input_stream, bed_line)) {
        // Remove whitespace from end of the line
        utils::rtrim(bed_line);
        if (bed_line.empty() || is_header(bed_line)) {
            continue;
        }

        std::string genome;
        BedFile::Entry entry;
        if (!try_get_entry_from_bedline(std::move(bed_line), genome, entry)) {
            spdlog::error("Invalid data reading bed file '{}'", m_file_name);
            return false;
        }
        m_genomes[genome].push_back(std::move(entry));
    }

    return true;
}

const BedFile::Entries& BedFile::entries(const std::string& genome) const {
    auto it = m_genomes.find(genome);
    return it != m_genomes.end() ? it->second : NO_ENTRIES;
}

}  // namespace dorado::alignment
