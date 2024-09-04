#include "fastq_reader.h"

#include <algorithm>
#include <cassert>
#include <fstream>

namespace dorado::utils {

namespace {

bool is_valid_id_field(const std::string& field) {
    if (field.at(0) != '@') {
        return false;
    }
    if (field.size() < 2 || field.at(1) == ' ') {
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

std::size_t token_len(const std::string& header_line, std::size_t token_start_pos) {
    auto token_end_pos = header_line.find(' ', token_start_pos);
    if (token_end_pos == std::string::npos) {
        token_end_pos = header_line.size();
    }
    return token_end_pos - token_start_pos;
}

}  // namespace

const std::string& FastqRecord::header() const { return m_header; }
const std::string& FastqRecord::sequence() const { return m_sequence; }
const std::string& FastqRecord::qstring() const { return m_qstring; }

std::string_view read_id_view(const std::string& header_line) {
    assert(header_line.size() > 1);
    return {header_line.data() + 1, token_len(header_line, 1)};
}

std::string_view run_id_view(const std::string& header_line) {
    // Fastq header line format:
    // @{read_id} runid={run_id} sampleid={sample_id} read={read_number} ch={channel_id} start_time={start_time_utc}
    const std::string RUN_ID_KEY_SEARCH{" runid="};
    auto runid_start = header_line.find(RUN_ID_KEY_SEARCH);
    if (runid_start == std::string::npos) {
        return {};
    }
    runid_start = runid_start + RUN_ID_KEY_SEARCH.size();

    return {header_line.data() + runid_start, token_len(header_line, runid_start)};
}

bool FastqRecord::set_id(std::string line) {
    if (!is_valid_id_field(line)) {
        return false;
    }
    m_header = std::move(line);
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
