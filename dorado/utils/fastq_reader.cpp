#include "fastq_reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace dorado::utils {

namespace {

bool is_valid_id_field(const std::string& field) {
    if (field.at(0) != '@') {
        return false;
    }

    std::stringstream field_stream{field};
    std::string id_section{};
    if (!std::getline(field_stream, id_section, ' ')) {
        return false;
    }
    if (id_section.size() < 2) {
        return false;
    }

    return true;
}

bool is_valid_sequence_field(std::string& field) {
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

bool set_id(FastqRecord& record, std::string line) {
    if (!is_valid_id_field(line)) {
        return false;
    }
    record.id = std::move(line);
    return true;
}

bool set_sequence(FastqRecord& record, std::string line) {
    if (!is_valid_sequence_field(line)) {
        return false;
    }
    record.sequence = std::move(line);
    return true;
}

bool set_separator(FastqRecord& record, std::string line) {
    if (!is_valid_separator_field(line)) {
        return false;
    }
    record.separator = std::move(line);
    return true;
}

bool set_quality(FastqRecord& record, std::string line) {
    if (!is_valid_quality_field(line)) {
        return false;
    }
    record.quality = std::move(line);
    return true;
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
    if (!get_non_empty_line(input_stream, line) || !set_id(result, std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !set_sequence(result, std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !set_separator(result, std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !set_quality(result, std::move(line))) {
        return std::nullopt;
    }

    if (result.sequence.size() != result.quality.size()) {
        return std::nullopt;
    }

    return result;
}

}  // namespace

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
