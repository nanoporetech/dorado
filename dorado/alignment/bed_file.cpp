#include "bed_file.h"

#include "utils/PostCondition.h"
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

bool get_tokens(const std::string& bed_line, std::vector<std::string>& tokens) {
    tokens.clear();
    std::istringstream bed_line_stream(bed_line);

    // if too many columns include an extra column so caller can validate and find the error
    for (std::size_t col{0}; col <= MAX_COLS; ++col) {
        std::string token;
        if (!std::getline(bed_line_stream, token, '\t')) {
            break;
        }
        if (token.empty()) {
            return false;
        }
        tokens.push_back(std::move(token));
    }

    return !tokens.empty();
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

bool is_header(const std::string& candidate) {
    return utils::starts_with(candidate, "browser") || utils::starts_with(candidate, "track");
}

bool is_comment(const std::string& candidate) {
    return candidate.empty() || candidate.at(0) == '#';
}

class BedFileEntryParser {
private:
    std::string m_candidate_bed_line;
    BedFile::Entry m_entry;
    std::string m_genome;
    std::string m_error_reason;
    std::vector<std::string> m_tokens;
    bool m_is_comment_line;
    std::size_t m_columns_per_entry{};
    bool m_in_header_section{true};
    bool m_is_valid{true};

    void reset(std::string bed_line) {
        m_candidate_bed_line = std::move(bed_line);
        utils::rtrim(m_candidate_bed_line);
        m_is_comment_line = is_comment(m_candidate_bed_line);
        m_entry = {};
        m_genome = {};
        m_error_reason = {};
        m_is_valid = true;
        m_tokens.clear();
    }

    bool validate_consistent_num_columns() {
        if (m_columns_per_entry == 0) {
            m_columns_per_entry = m_tokens.size();
            return true;
        }

        if (m_tokens.size() == m_columns_per_entry) {
            return true;
        }

        std::ostringstream oss{};
        oss << "Inconsistent number of columns. Expected: " << m_columns_per_entry
            << " actual: " << m_tokens.size() << ".";
        m_error_reason = oss.str();
        return false;
    }

    bool try_load_tokens() {
        if (!get_tokens(m_candidate_bed_line, m_tokens)) {
            m_error_reason = "Missing columns.";
            return false;
        }

        if (!validate_consistent_num_columns()) {
            return false;
        }

        const auto num_columns = m_tokens.size();
        if (num_columns < MIN_COLS) {
            m_error_reason = "Too few columns (minimum 3).";
            return false;
        }
        if (num_columns > MAX_COLS) {
            m_error_reason = "Too many columns (maximum 12).";
            return false;
        }

        return true;
    }

    bool try_process_tokens() {
        BedFile::Entry new_entry{};
        if (!try_get(m_tokens[1], new_entry.start)) {
            m_error_reason = "Unable to read column 2: [START]";
            return false;
        }

        if (!try_get(m_tokens[2], new_entry.end)) {
            m_error_reason = "Unable to read column 3: [END].";
            return false;
        }

        if (m_tokens.size() > 5 && !try_get_strand(m_tokens[5], new_entry.strand)) {
            m_error_reason = "Unable to read column 6: [STRAND].";
            return false;
        }

        m_genome = m_tokens[0];
        m_entry = std::move(new_entry);
        m_entry.bed_line = std::move(m_candidate_bed_line);

        return true;
    }

    void append_bed_line_to_err_message() {
        if (m_is_valid) {
            return;
        }
        std::ostringstream oss;
        oss << m_error_reason << " LINE[" << std::quoted(m_candidate_bed_line) << "]";
        m_error_reason = oss.str();
    }

    bool is_in_header_section() {
        if (!m_in_header_section) {
            return false;
        }
        if (is_header(m_candidate_bed_line)) {
            return true;
        }
        m_in_header_section = false;
        return false;
    }

public:
    BedFile::Entry& entry() { return m_entry; }
    const std::string& genome() const { return m_genome; };
    const std::string& error_reason() const { return m_error_reason; }

    bool ignore_line() const { return m_is_comment_line || m_in_header_section; }

    bool is_valid() const { return m_is_valid; }

    bool parse(std::string bed_line) {
        reset(std::move(bed_line));

        if (m_is_comment_line || is_in_header_section()) {
            return true;
        }

        auto ensure_err_msg_has_bed_line =
                utils::PostCondition([this] { append_bed_line_to_err_message(); });

        if (!try_load_tokens() || !try_process_tokens()) {
            m_is_valid = false;
        }
        return m_is_valid;
    }
};

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
    BedFileEntryParser parser{};
    std::size_t line_number{1};
    while (std::getline(input_stream, bed_line) && parser.parse(std::move(bed_line))) {
        ++line_number;
        if (parser.ignore_line()) {
            continue;
        }
        m_genomes[parser.genome()].push_back(std::move(parser.entry()));
    }

    if (!parser.is_valid()) {
        spdlog::error("Invalid data reading bed file '{}' at line {}. {}", m_file_name, line_number,
                      parser.error_reason());
        return false;
    }

    return true;
}

const BedFile::Entries& BedFile::entries(const std::string& genome) const {
    auto it = m_genomes.find(genome);
    return it != m_genomes.end() ? it->second : NO_ENTRIES;
}

}  // namespace dorado::alignment
