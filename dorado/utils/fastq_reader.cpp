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
            spdlog::warn("Error closing hFILE. ErrNo: {}.", errno);
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

bool is_valid_id_field(const std::string& field) {
    if (field.size() < 2 || field.at(0) != '@') {
        return false;
    }

    const auto id_start_char = field.at(1);
    if (id_start_char == ' ' || id_start_char == '\t') {
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

bool is_valid_separator_field(const std::string& field) {
    assert(!field.empty());
    return field.at(0) == '+';
}

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

char header_separator(bool has_bam_tags) { return has_bam_tags ? '\t' : ' '; }

void ignore_next_tab_separated_field(std::istringstream& header_stream) {
    header_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\t');
}

}  // namespace

const std::string& FastqRecord::header() const { return m_header; }
const std::string& FastqRecord::sequence() const { return m_sequence; }
const std::string& FastqRecord::qstring() const { return m_qstring; }

std::size_t FastqRecord::token_len(std::size_t token_start_pos) const {
    const auto separator = header_separator(m_header_has_bam_tags);
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

bool FastqRecord::set_header(std::string line) {
    // Fastq header line formats that we currently recognise beyond the initial @{read_id} are
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
