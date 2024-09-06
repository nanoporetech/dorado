#include "fastq_reader.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <limits>
#include <sstream>

namespace dorado::utils {

namespace {

bool is_valid_id_field(const std::string& field) {
    if (field.at(0) != '@') {
        return false;
    }
    if (field.size() < 2 || field.at(1) == ' ' || field.at(1) == '\t') {
        return false;
    }
    return true;
}

bool validate_sequence_and_replace_us(std::string& field) {
    bool contains_t{};
    bool contains_u{};
    for (auto& element : field) {
        switch (element) {
        case 'A':
        case 'C':
        case 'G':
            break;
        case 'T':
            if (contains_u) {
                return false;
            }
            contains_t = true;
            break;
        case 'U':
            if (contains_t) {
                return false;
            }
            contains_u = true;
            element = 'T';
            break;
        default:
            return false;
        }
    }

    return true;
}

bool is_valid_separator_field(const std::string& field) { return field == "+"; }

bool is_valid_quality_field(const std::string& field) {
    //0x21 (lowest quality; '!' in ASCII) to 0x7e (highest quality; '~' in ASCII)
    return std::none_of(field.begin(), field.end(), [](char c) { return c < 0x21 || c > 0x7e; });
}

bool get_non_empty_line(std::istream& input_stream, std::string& line) {
    if (!std::getline(input_stream, line)) {
        return false;
    }
    return !line.empty();
}

std::optional<FastqRecord> get_next_record(std::istream& input_stream) {
    if (!input_stream.good()) {
        return std::nullopt;
    }
    FastqRecord result;
    std::string line;
    if (!get_non_empty_line(input_stream, line) || !result.set_id(std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !result.set_sequence(std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !is_valid_separator_field(line)) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !result.set_quality(std::move(line))) {
        return std::nullopt;
    }

    if (result.sequence().size() != result.qstring().size()) {
        return std::nullopt;
    }

    return result;
}

char header_separator(bool has_bam_tags) { return has_bam_tags ? '\t' : ' '; }

void ignore_next_tab_separated_field(std::istringstream& header_stream) {
    header_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\t');
}

}  // namespace

const std::string& FastqRecord::header() const { return m_header; }
const std::string& FastqRecord::sequence() const { return m_sequence; }
const std::string& FastqRecord::qstring() const { return m_qstring; }

std::size_t FastqRecord::token_len(std::size_t token_start_pos) const {
    auto separator = header_separator(m_header_has_bam_tags);
    auto token_end_pos = m_header.find(separator, token_start_pos);
    if (token_end_pos == std::string::npos) {
        token_end_pos = m_header.size();
    }
    return token_end_pos - token_start_pos;
}

std::string_view FastqRecord::read_id_view() const {
    assert(m_header.size() > 1);
    return {m_header.data() + 1, token_len(1)};
}

std::string_view FastqRecord::run_id_view() const {
    if (m_header_has_bam_tags) {
        return {};  // HtsLib style
    }
    // Assume minKNOW format and check for the runid key
    const std::string RUN_ID_KEY_SEARCH{" runid="};
    auto runid_start = m_header.find(RUN_ID_KEY_SEARCH);
    if (runid_start == std::string::npos) {
        return {};
    }
    runid_start = runid_start + RUN_ID_KEY_SEARCH.size();

    return {m_header.data() + runid_start, token_len(runid_start)};
}

std::vector<std::string> FastqRecord::get_bam_tags() const {
    if (!m_header_has_bam_tags) {
        return {};
    }
    std::vector<std::string> result{};
    std::istringstream header_stream{m_header};

    // First field is the read ID not a bam tag
    ignore_next_tab_separated_field(header_stream);

    std::string tag;
    while (std::getline(header_stream, tag, '\t')) {
        result.push_back(std::move(tag));
    }
    return result;
}

bool FastqRecord::set_id(std::string line) {
    // Fastq header line format we currently recognise beyond the initial @{read_id} field are
    // a) minKNOW style:
    // @{read_id} runid={run_id} sampleid={sample_id} read={read_number} ch={channel_id} start_time={start_time_utc}
    // or,
    // b) HtsLib, which embeds tab separated bam tags:
    // @{read_id}['\t'{tag_data}...]
    //
    // Other formats should be of the form @{read_id}[ {description}]
    if (!is_valid_id_field(line)) {
        return false;
    }
    m_header = std::move(line);
    if (m_header.find('\t') != std::string::npos) {
        m_header_has_bam_tags = true;
    }
    return true;
}

bool FastqRecord::set_sequence(std::string line) {
    if (!validate_sequence_and_replace_us(line)) {
        return false;
    }
    m_sequence = std::move(line);
    return true;
}

bool FastqRecord::set_quality(std::string line) {
    if (!is_valid_quality_field(line)) {
        return false;
    }
    m_qstring = std::move(line);
    return true;
}

FastqReader::FastqReader(const std::string& input_file) {
    if (!is_fastq(input_file)) {
        return;
    }
    m_input_stream = std::make_unique<std::ifstream>(input_file);
}

FastqReader::FastqReader(std::unique_ptr<std::istream> input_stream) {
    if (!is_fastq(*input_stream)) {
        return;
    }
    // return to start of stream after validating the first record.
    input_stream->clear();
    input_stream->seekg(0);
    m_input_stream = std::move(input_stream);
}

bool FastqReader::is_valid() const { return m_input_stream && m_input_stream->good(); }

std::optional<FastqRecord> FastqReader::try_get_next_record() {
    if (!m_input_stream) {
        return std::nullopt;
    }
    return get_next_record(*m_input_stream);
}

bool is_fastq(const std::string& input_file) {
    std::ifstream input_stream{input_file};
    return is_fastq(input_stream);
}

bool is_fastq(std::istream& input_stream) {
    if (!input_stream.good()) {
        return false;
    }

    return get_next_record(input_stream).has_value();
}

}  // namespace dorado::utils
