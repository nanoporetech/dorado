#include "fasta_reader.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <limits>
#include <sstream>

namespace dorado::utils {

namespace {

bool get_wrapped_sequence_line(std::istream& input_stream, std::string& wrapped_line) {
    std::string line;
    std::ostringstream line_builder{};
    while (input_stream.peek() != '>') {
        if (!get_non_empty_line(input_stream, line)) {
            break;
        }
        if (!validate_sequence_and_replace_us(line)) {
            return false;
        }
        line_builder << line;
    }
    wrapped_line = line_builder.str();
    return !wrapped_line.empty();
}

}  // namespace

std::string FastaRecord::record_name() const {
    const auto& tokens = m_header.get_tokens();
    assert(!tokens.empty() && tokens[0].size() > 1);
    return tokens[0].substr(1);
}

const std::string& FastaRecord::header() const { return m_header.header(); }

const std::string& FastaRecord::sequence() const { return m_sequence; }

bool FastaRecord::set_header(std::string line) { return m_header.set_header(std::move(line)); }

std::optional<FastaRecord> FastaRecord::try_create(std::istream& input_stream,
                                                   std::string& error_message) {
    if (!input_stream.good()) {
        return std::nullopt;
    }
    FastaRecord result;
    std::string line;
    if (!get_non_empty_line(input_stream, line)) {
        return std::nullopt;
    }
    if (!result.set_header(std::move(line))) {
        error_message = "Invalid header line.";
        return std::nullopt;
    }
    if (!get_wrapped_sequence_line(input_stream, line)) {
        error_message = "Invalid sequence.";
        return std::nullopt;
    }
    result.m_sequence = std::move(line);
    return result;
}

FastaReader::FastaReader(std::string input_file) : m_input_file(std::move(input_file)) {
    if (!is_fasta(m_input_file)) {
        return;
    }
    m_input_stream = std::make_unique<std::ifstream>(m_input_file);
}

FastaReader::FastaReader(std::unique_ptr<std::istream> input_stream)
        : m_input_file("<input_stream>") {
    if (!is_fasta(*input_stream)) {
        return;
    }
    // return to start of stream after validating the first record.
    input_stream->clear();
    input_stream->seekg(0);
    m_input_stream = std::move(input_stream);
}

bool FastaReader::is_valid() const { return m_input_stream && m_input_stream->good(); }

std::optional<FastaRecord> FastaReader::try_get_next_record() {
    if (!m_input_stream) {
        return std::nullopt;
    }
    ++m_record_count;
    std::string error_message{};
    auto next_fasta_record = FastaRecord::try_create(*m_input_stream, error_message);
    if (!error_message.empty()) {
        spdlog::warn("Failed to read record #{} from {}. {}", m_record_count, m_input_file,
                     error_message);
    }

    return next_fasta_record;
}

bool is_fasta(const std::string& input_file) {
    std::ifstream input_stream{input_file};
    return is_fasta(input_stream);
}

bool is_fasta(std::istream& input_stream) {
    if (!input_stream.good()) {
        return false;
    }

    std::string ignore_error_when_checking;
    return FastaRecord::try_create(input_stream, ignore_error_when_checking).has_value();
}

}  // namespace dorado::utils
