#include "fastq_reader.h"

#include "gzip_reader.h"
#include "types.h"

#include <htslib/hfile.h>
#include <htslib/hts.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <limits>
#include <sstream>

namespace dorado::utils {

namespace {

class GzipStreamBuf : public std::streambuf {
    GzipReader m_gzip_reader;

    void throw_to_set_bad_bit_if_gzip_reader_invalid() {
        // throwing during a call to underflow will cause the stream bad bit to be set
        if (m_gzip_reader.is_valid()) {
            return;
        }
        spdlog::error(m_gzip_reader.error_message());
        throw std::runtime_error(m_gzip_reader.error_message());
    }

    void read_next_chunk_into_get_area() {
        while (m_gzip_reader.read_next() && m_gzip_reader.is_valid() &&
               m_gzip_reader.num_bytes_read() == 0) {
        }
        throw_to_set_bad_bit_if_gzip_reader_invalid();
        setg(m_gzip_reader.decompressed_buffer().data(), m_gzip_reader.decompressed_buffer().data(),
             m_gzip_reader.decompressed_buffer().data() + m_gzip_reader.num_bytes_read());
    }

public:
    GzipStreamBuf(const std::string& gzip_file, std::size_t buffer_size)
            : m_gzip_reader(gzip_file, buffer_size) {}

    int underflow() {
        throw_to_set_bad_bit_if_gzip_reader_invalid();
        if (gptr() == egptr()) {
            read_next_chunk_into_get_area();
        }
        return gptr() == egptr() ? std::char_traits<char>::eof()
                                 : std::char_traits<char>::to_int_type(*gptr());
    }
};

class GzipInputStream : public std::istream {
    std::unique_ptr<GzipStreamBuf> m_gzip_stream_buf{};

public:
    GzipInputStream(const std::string& gzip_file, std::size_t buffer_size)
            : std::istream(nullptr),
              m_gzip_stream_buf(std::make_unique<GzipStreamBuf>(gzip_file, buffer_size)) {
        // The base class (istream) will be constructed first so can't pass
        // the buffer as part of the member initalisation list, instead set
        // the buffer here after the base class has been constructed.
        rdbuf(m_gzip_stream_buf.get());
    }
};

bool check_file_can_be_opened_for_reading(const std::string& input_file) {
    std::ifstream check_normal_file(input_file);
    return check_normal_file.is_open();
}

struct HfileDestructor {
    void operator()(hFILE* fp) {
        if (fp && hclose(fp) == EOF) {
            spdlog::warn("Problem closing hFILE. return code: {}.", errno);
        }
    }
};
using HfilePtr = std::unique_ptr<hFILE, HfileDestructor>;

std::unique_ptr<std::istream> create_input_stream(const std::string& input_file) {
    // check for a normal file that can be opened for reading before calling hopen
    // as hopen has special semantics and does not do this check, e.g. calling with
    // "-" would block waiting on the stdin stream.
    if (!check_file_can_be_opened_for_reading(input_file)) {
        return {};
    }
    HfilePtr hfile(hopen(input_file.c_str(), "r"));
    if (!hfile) {
        return {};
    }
    htsFormat format_check;
    auto fmt_detect_result = hts_detect_format(hfile.get(), &format_check);
    // Note the format check does not detect fastq if Ts are replaced with Us
    // So treat a text file as a potential fastq
    if (fmt_detect_result < 0 || (format_check.format != htsExactFormat::fastq_format &&
                                  format_check.format != htsExactFormat::text_format)) {
        return {};
    }

    static constexpr std::size_t DECOMPRESSION_BUFFER_SIZE{65536};
    if (format_check.compression == htsCompression::no_compression) {
        return std::make_unique<std::ifstream>(input_file);
    } else if (format_check.compression == htsCompression::gzip) {
        return std::make_unique<GzipInputStream>(input_file, DECOMPRESSION_BUFFER_SIZE);
    }

    return {};
}

bool is_valid_separator_field(const std::string& field) {
    assert(!field.empty());
    return field.at(0) == '+';
}

bool is_valid_quality_field(const std::string& field) {
    //0x21 (lowest quality; '!' in ASCII) to 0x7e (highest quality; '~' in ASCII)
    return std::none_of(field.begin(), field.end(), [](char c) { return c < 0x21 || c > 0x7e; });
}

bool get_wrapped_qstring_line(std::istream& input_stream,
                              std::size_t sequence_size,
                              std::string& wrapped_line) {
    std::string line;
    std::ostringstream line_builder{};
    std::size_t qstring_size{};
    while (qstring_size < sequence_size && get_non_empty_line(input_stream, line)) {
        if (!is_valid_quality_field(line)) {
            return false;
        }
        qstring_size += line.size();
        if (qstring_size > sequence_size) {
            return false;
        }
        line_builder << line;
    }
    wrapped_line = line_builder.str();
    return wrapped_line.size() == sequence_size;
}

bool get_wrapped_sequence_line(std::istream& input_stream, std::string& wrapped_line) {
    std::string line;
    std::ostringstream line_builder{};
    while (input_stream.peek() != '+') {
        if (!get_non_empty_line(input_stream, line) || !validate_sequence_and_replace_us(line)) {
            return false;
        }
        line_builder << line;
    }
    wrapped_line = line_builder.str();
    return !wrapped_line.empty();
}

}  // namespace

const std::string& FastqRecord::header() const { return m_header.header(); }
const std::string& FastqRecord::sequence() const { return m_sequence; }
const std::string& FastqRecord::qstring() const { return m_qstring; }

std::string_view FastqRecord::read_id_view() const {
    const auto& tokens = m_header.get_tokens();
    assert(!tokens.empty() && tokens[0].size() > 1);
    return std::string_view(tokens[0]).substr(1);
}

std::string_view FastqRecord::run_id_view() const {
    if (m_header.has_bam_tags()) {
        return {};  // HtsLib style
    }
    // Assume minKNOW format and check for the runid key
    const std::string RUN_ID_KEY_SEARCH{"runid="};
    const auto& tokens = m_header.get_tokens();
    if (tokens.size() < 2) {
        return {};
    }
    for (const auto& token : tokens) {
        auto token_view = std::string_view(token);
        if (token_view.substr(0, RUN_ID_KEY_SEARCH.size()) == RUN_ID_KEY_SEARCH) {
            return token_view.substr(RUN_ID_KEY_SEARCH.size());
        }
    }
    return {};
}

std::vector<std::string> FastqRecord::get_bam_tags() const {
    if (!m_header.has_bam_tags()) {
        return {};
    }
    const auto& tokens = m_header.get_tokens();
    if (tokens.size() < 2) {
        return {};
    }
    std::vector<std::string> tags{};
    auto iter = tokens.begin();
    ++iter;
    tags.insert(tags.end(), iter, tokens.end());
    return tags;
}

bool FastqRecord::set_header(std::string line) { return m_header.set_header(std::move(line)); }

std::optional<FastqRecord> FastqRecord::try_create(std::istream& input_stream,
                                                   std::string& error_message) {
    if (!input_stream.good()) {
        return std::nullopt;
    }
    FastqRecord result;
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
    if (!get_non_empty_line(input_stream, line) || !is_valid_separator_field(line)) {
        error_message = "Invalid separator.";
        return std::nullopt;
    }
    if (!get_wrapped_qstring_line(input_stream, result.sequence().size(), line)) {
        error_message = "Invalid qstring.";
        return std::nullopt;
    }
    result.m_qstring = std::move(line);

    return result;
}

bool operator==(const FastqRecord& lhs, const FastqRecord& rhs) {
    return std::tie(lhs.header(), lhs.sequence(), lhs.qstring()) ==
           std::tie(rhs.header(), rhs.sequence(), rhs.qstring());
}

bool operator!=(const FastqRecord& lhs, const FastqRecord& rhs) { return !(lhs == rhs); }

FastqReader::FastqReader(std::string input_file) : m_input_file(std::move(input_file)) {
    if (!is_fastq(m_input_file)) {
        return;
    }

    // Simplest to create another input stream after the is_fastq check because our
    // gzip istream has not implemented seek.
    m_input_stream = create_input_stream(m_input_file);
}

FastqReader::FastqReader(std::unique_ptr<std::istream> input_stream)
        : m_input_file("<input_stream>") {
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
    ++m_record_count;
    std::string error_message{};
    auto next_fastq_record = FastqRecord::try_create(*m_input_stream, error_message);
    if (!error_message.empty()) {
        spdlog::warn("Failed to read record #{} from {}. {}", m_record_count, m_input_file,
                     error_message);
    }

    return next_fastq_record;
}

bool is_fastq(const std::string& input_file) {
    auto input_stream = create_input_stream(input_file);
    return input_stream ? is_fastq(*input_stream) : false;
}

bool is_fastq(std::istream& input_stream) {
    if (!input_stream.good()) {
        return false;
    }

    std::string ignore_error_when_checking;
    return FastqRecord::try_create(input_stream, ignore_error_when_checking).has_value();
}

}  // namespace dorado::utils
