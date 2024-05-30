#include "sam_utils.h"

#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace {

class SamLineStream : public std::istringstream {
    template <typename T>
    using enable_if_integral_t = typename std::enable_if<
            std::is_integral<typename std::remove_reference<T>::type>::value>::type;
    template <typename T>
    using enable_if_string_t = typename std::enable_if<
            std::is_same<typename std::remove_reference<T>::type, std::string>::value>::type;

    template <typename T, enable_if_integral_t<T>* = nullptr>
    T default_value() {
        return 0;
    }

    template <typename T, enable_if_string_t<T>* = nullptr>
    T default_value() {
        return "*";
    }

public:
    using std::istringstream::istringstream;

    template <typename T>
    SamLineStream& operator>>(T& val) {
        if (peek() == '\t') {
            spdlog::warn("Empty sam line field in stream. Continuing anyway.");
            val = default_value<T>();
        } else {
            dynamic_cast<std::istringstream&>(*this) >> val;
        }
        get();
        return *this;
    }
};

std::pair<char, int> next_op(const char*& seq) {
    const char* begin = seq;
    int read_number = 0;
    while (seq && *seq >= '0' && *seq <= '9') {
        read_number = read_number * 10 + int(*(seq++) - '0');
    }
    if (!seq || !*seq || begin == seq) {
        return std::make_pair('?', -1);
    }
    return std::make_pair(*(seq++), read_number);
}

}  // namespace

namespace dorado::alignment {

int parse_cigar(const std::string& cigar, dorado::AlignmentResult& result) {
    const char* curr = cigar.c_str();
    const char* end = curr + cigar.length();
    bool first = true;
    result.strand_start = 0;
    result.num_insertions = 0;
    result.num_deletions = 0;
    result.num_aligned = 0;
    char type;
    int length;
    int hard_clipped = 0;
    while (curr != end) {
        std::tie(type, length) = next_op(curr);
        if (type == 'H') {
            type = 'S';
            hard_clipped += length;
        }
        switch (type) {
        case 'S':
            if (first) {
                result.strand_start = length;
            }
            break;
        case 'I':
            result.num_insertions += length;
            break;
        case 'D':
            result.num_deletions += length;
            break;
        case 'M':
            result.num_aligned += length;
            break;
        default:
            throw std::runtime_error("Currently only supporting HSIDM in SAM cigar string.");
        }
        first = false;
    }
    return hard_clipped;
}

std::vector<AlignmentResult> parse_sam_lines(const std::string& sam_content,
                                             const std::string& query_seq,
                                             const std::string& query_qual) {
    std::vector<AlignmentResult> results;
    std::istringstream sam_content_stream(sam_content);
    std::unordered_map<std::string, int> reference_length;
    // read header
    std::string header_line;
    std::string sq, reference, length_field;
    while (char(sam_content_stream.peek()) == '@') {
        // Read the genome sequence lengths from the header
        sam_content_stream >> sq >> reference >> length_field;
        if (sq == "@SQ") {
            spdlog::trace(__func__ + std::string{"length_field: {}"}, length_field);
            int ref_length = std::stoi(length_field.substr(3, length_field.size()));
            reference_length[reference.substr(3, reference.size())] = ref_length;
        }
        std::getline(sam_content_stream, header_line);
    }

    // Read every alignment from the SAM file
    for (std::string sam_line; std::getline(sam_content_stream, sam_line);) {
        AlignmentResult res{};

        // required fields
        std::string seq_name, cigar, rnext, aligned_seq, aligned_qstring;
        unsigned int flags;
        int map_quality, next_pos, seq_len;
        SamLineStream sam_line_istream(sam_line);
        sam_line_istream             // SAM column
                >> seq_name          // 1 QNAME
                >> flags             // 2 FLAG
                >> res.genome        // 3 RNAME
                >> res.genome_start  // 4 POS
                >> map_quality       // 5 MAPQ
                >> cigar             // 6 CIGAR
                >> rnext             // 7 RNEXT
                >> next_pos          // 8 PNEXT
                >> seq_len           // 9 TLEN
                >> aligned_seq       // 10 SEQ
                >> aligned_qstring   // 11 QUAL
                ;

        // If the output sam stream did not contain a SEQ, we try to fill it
        //  in with the passed in query_seq and query_qual
        if ((aligned_seq == "*" || aligned_seq.empty()) && !query_seq.empty()) {
            if (flags & 0x10u) {  // Strand should be reversed from query
                res.sequence = utils::reverse_complement(query_seq);
            } else {
                res.sequence = query_seq;
            }

            res.qstring = query_qual;
            if (flags & 0x10u) {  // qstring should be reversed from query
                std::reverse(res.qstring.begin(), res.qstring.end());
            }
        } else {
            // There was an aligned_seq so it should be used (note it may have been hard-clipped by minimap)
            res.sequence = aligned_seq;

            // We may still need to replace the qstring in the output with a valid qstring if appropriate.
            // If the qstring is missing from the returned line, and the supplied qstring is the right length
            // (no hard clipping has taken place), we drop it in.
            if ((aligned_qstring == "*" || aligned_qstring.empty()) &&
                aligned_seq.length() == query_qual.length()) {
                res.qstring = query_qual;
                if (flags & 0x10u) {  // qstring should be reversed from query
                    std::reverse(res.qstring.begin(), res.qstring.end());
                }
            } else {
                res.qstring = aligned_qstring;
            }
        }

        res.num_events = int(res.sequence.length());
        res.mapping_quality = map_quality;

        typedef struct {
            std::string Type, Value;
        } TypeValuePair;

        std::map<std::string, TypeValuePair> opt_values;

        if (res.genome != "*") {
            // optional fields
            while (!sam_line_istream.eof()) {
                std::string field;
                sam_line_istream >> field;
                if (field.length() < 5 || field[2] != ':' || field[4] != ':') {
                    throw std::runtime_error("optional SAM field '" + field +
                                             "' could not be parsed.");
                }
                std::string key = field.substr(0, 2);
                char type = field[3];
                std::string value = field.substr(5);
                opt_values[key] = TypeValuePair{std::string(1, type), value};
            }

            // write to output structure
            if (cigar == "*") {
                res.coverage =
                        float(seq_len) / float(std::min(seq_len, reference_length[res.genome]));
                res.genome_end = std::min(res.genome_start + seq_len, reference_length[res.genome]);
                res.strand_end = res.strand_start + seq_len;
                res.num_correct = 0;
                res.identity = 0.0f;
                res.accuracy = 0.0f;
                res.strand_score = 0;
            } else {
                int hard_clipped = parse_cigar(cigar, res);
                int full_len = res.num_events;
                if (query_seq.empty()) {
                    full_len += hard_clipped;
                }
                res.coverage = float(res.num_aligned) /
                               float(std::min(full_len, reference_length[res.genome]));
                res.genome_end = res.genome_start + res.num_aligned + res.num_deletions;
                res.strand_end = res.strand_start + res.num_aligned + res.num_insertions;
                if (opt_values.find("NM") == opt_values.end()) {
                    throw std::runtime_error("Input SAM line for read ID " + seq_name +
                                             " does not contain required 'NM' tag");
                }
                spdlog::trace(__func__ + std::string{"opt_values.at(\"NM\").Value: {}"},
                              opt_values.at("NM").Value);
                int edit_distance = std::stoi(opt_values.at("NM").Value);
                int num_mismatches = edit_distance - res.num_insertions - res.num_deletions;
                res.num_correct = res.num_aligned - num_mismatches;
                res.identity = float(res.num_correct) / float(res.num_aligned);
                res.accuracy = float(res.num_correct) /
                               float(res.num_aligned + res.num_insertions + res.num_deletions);
                if (opt_values.find("AS") != opt_values.end()) {
                    spdlog::trace(__func__ + std::string{"opt_values.at(\"AS\").Value: {}"},
                                  opt_values.at("AS").Value);
                    res.strand_score = std::stoi(opt_values.at("AS").Value);
                } else {
                    res.strand_score = 0;
                }
            }
            res.direction = (flags & 0x10u) ? '-' : '+';
        }
        res.name = seq_name;
        res.secondary_alignment = ((flags & 0x100u) != 0);
        res.supplementary_alignment = ((flags & 0x800u) != 0);

        // Rebuild the SAM line, since it may have been malformed, and is probably missing qual data.
        std::ostringstream sam_line_ostream;
        sam_line_ostream << seq_name << '\t' << flags << '\t' << res.genome << '\t'
                         << res.genome_start << '\t' << map_quality << '\t' << cigar << '\t'
                         << rnext << '\t' << next_pos << '\t' << seq_len << '\t' << res.sequence
                         << '\t' << res.qstring;
        if (!opt_values.empty()) {
            for (const auto& item : opt_values) {
                sam_line_ostream << '\t' << item.first << ':' << item.second.Type << ':'
                                 << item.second.Value;
            }
        }
        res.sam_string = sam_line_ostream.str();
        results.push_back(res);
    }

    // We want to make sure the primary alignment is first.
    auto is_primary_alignment = [](const AlignmentResult& res) {
        return !(res.secondary_alignment || res.supplementary_alignment);
    };
    auto primary_alignment = std::find_if(results.begin(), results.end(), is_primary_alignment);
    if (primary_alignment != results.end()) {
        std::iter_swap(primary_alignment, results.begin());
    }
    return results;
}

}  // namespace dorado::alignment
