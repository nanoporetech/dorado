#pragma once

#include "fasta_fastq_utils.h"

#include <istream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dorado::utils {

class FastqRecord {
public:
    FastqRecord() : m_header(FastaFastqHeader::FASTQ) {}

    // The header will be at least 2 characters and begin with '@'
    const std::string& header() const;

    // All characters will be in the set {A,C,G,T}
    const std::string& sequence() const;

    // All characters will be in the range of printable characters ['!' .. '~'],
    // i.e. [33 .. 126]
    const std::string& qstring() const;

    std::string_view read_id_view() const;

    // Retrieves the run_id from the header if it has been embedded in minKNOW style format
    // or an enpty string if not found.
    // e.g.
    // @read_0 runid=123 sampleid=... etc.
    // would return "123"
    std::string_view run_id_view() const;

    // Retrieves list of bam tags if they have been embedded in the header.
    // When dorado uses HtsLib to output to fastq some tags will be embedded in
    // the header these will be returned in string format e.g.
    // result[0]: "st:Z:2023-06-22T07:17:48.308+00:00"
    // result[1]: "RG:Z:6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4.3.0"
    // etc ...
    std::vector<std::string> get_bam_tags() const;

    bool set_header(std::string value);

    static std::optional<FastqRecord> try_create(std::istream& input_stream,
                                                 std::string& error_message);

private:
    FastaFastqHeader m_header;
    std::string m_sequence;
    std::string m_qstring;
};

bool operator==(const FastqRecord& lhs, const FastqRecord& rhs);
bool operator!=(const FastqRecord& lhs, const FastqRecord& rhs);

class FastqReader {
public:
    FastqReader(const std::string input_file);
    FastqReader(std::unique_ptr<std::istream> input_stream);
    bool is_valid() const;
    std::optional<FastqRecord> try_get_next_record();

private:
    std::string m_input_file;
    std::unique_ptr<std::istream> m_input_stream{};
    std::size_t m_record_count{};
};

// Check for a fastq file. Does basic checks on the four fields of the first record
// will return true for a sequence containing Us instead of Ts, so if the check
// succeeds it is still possible the file cannot be opened by HtsLib
bool is_fastq(const std::string& input_file);
bool is_fastq(std::istream& input_stream);

}  // namespace dorado::utils
