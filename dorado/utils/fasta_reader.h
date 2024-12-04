#pragma once

#include "fasta_fastq_utils.h"

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::utils {

class FastaRecord {
public:
    FastaRecord() : m_header(FastaFastqHeader::FASTA) {}

    // The header will be at least 2 characters and begin with '>'
    const std::string& header() const;

    std::string record_name() const;

    // All characters will be in the set {A,C,G,T}
    const std::string& sequence() const;

    const std::vector<std::string>& get_tokens() const { return m_header.get_tokens(); }

    bool set_header(std::string value);

    static std::optional<FastaRecord> try_create(std::istream& input_stream,
                                                 std::string& error_message);

private:
    FastaFastqHeader m_header;
    std::string m_sequence;
};

class FastaReader {
public:
    FastaReader(const std::string input_file);
    FastaReader(std::unique_ptr<std::istream> input_stream);
    bool is_valid() const;
    std::optional<FastaRecord> try_get_next_record();

private:
    std::string m_input_file;
    std::unique_ptr<std::istream> m_input_stream{};
    std::size_t m_record_count{};
};

// Check for a fasta file. Does basic checks on the two fields of the first record
// will return true for a sequence containing Us instead of Ts, so if the check
// succeeds it is still possible the file cannot be opened by HtsLib
bool is_fasta(const std::string& input_file);
bool is_fasta(std::istream& input_stream);

}  // namespace dorado::utils
