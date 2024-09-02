#pragma once

#include <istream>
#include <memory>
#include <optional>
#include <string>

namespace dorado::utils {

struct FastqRecord {
    std::string id;
    std::string sequence;
    std::string separator;
    std::string quality;
};

class FastqReader {
public:
    FastqReader(const std::string& input_file);
    FastqReader(std::unique_ptr<std::istream> input_stream);
    bool is_valid() const;
    std::optional<FastqRecord> try_get_next_record();

private:
    std::unique_ptr<std::istream> m_input_stream{};
};

// Check for a fastq file. Does basic checks on the four fields of the first record
// will return true for a sequence containing Us instead of Ts, so if the check
// succeeds it is still possible the file cannot be opened by HtsLib
bool is_fastq(const std::string& input_file);
bool is_fastq(std::istream& input_stream);

}  // namespace dorado::utils