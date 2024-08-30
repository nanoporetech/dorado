#include "fastq_reader.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
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

bool is_valid_sequence_field(const std::string& field) {
    bool contains_t{};
    bool contains_u{};
    for (const auto& element : field) {
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

struct FastqRecord {
    std::string id;
    std::string sequence;
    std::string separator;
    std::string quality;

    bool set_id(std::string line) {
        if (!is_valid_id_field(line)) {
            return false;
        }
        id = std::move(line);
        return true;
    }

    bool set_sequence(std::string line) {
        if (!is_valid_sequence_field(line)) {
            return false;
        }
        sequence = std::move(line);
        return true;
    }

    bool set_separator(std::string line) {
        if (!is_valid_separator_field(line)) {
            return false;
        }
        separator = std::move(line);
        return true;
    }

    bool set_quality(std::string line) {
        if (!is_valid_quality_field(line)) {
            return false;
        }
        quality = std::move(line);
        return true;
    }
};

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
    if (!get_non_empty_line(input_stream, line) || !result.set_separator(std::move(line))) {
        return std::nullopt;
    }
    if (!get_non_empty_line(input_stream, line) || !result.set_quality(std::move(line))) {
        return std::nullopt;
    }

    if (result.sequence.size() != result.quality.size()) {
        return std::nullopt;
    }

    return result;
}

}  // namespace

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
