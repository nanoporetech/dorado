#include "alignment/sam_utils.h"

#include "utils/SeparatedStream.h"
#include "utils/log_utils.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace dorado::alignment {

namespace {

class SamLineStream {
    utils::TabSeparatedStream m_stream;

    template <typename T>
    static T default_value() requires std::is_integral_v<T> {
        return 0;
    }

    template <typename T>
    static T default_value()
            requires(std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>) {
        return "*";
    }

public:
    SamLineStream(std::string_view line) : m_stream(line) {}

    template <typename T>
    SamLineStream& operator>>(T& val) {
        if (m_stream.peek().value_or("").empty()) {
            utils::trace_log("Empty sam line field in stream. Continuing anyway.");
            val = default_value<T>();
            // Consume the empty field.
            (void)m_stream.getline();
        } else {
            m_stream >> val;
        }
        return *this;
    }

    [[nodiscard]] operator bool() const { return m_stream; }
};

std::pair<char, int> next_op(std::string_view& seq) {
    // Look for the type.
    auto is_number = [](char c) { return '0' <= c && c <= '9'; };
    const auto type = std::find_if_not(seq.begin(), seq.end(), is_number);
    if (type == seq.begin() || type == seq.end()) {
        throw std::runtime_error("Bad CIGAR sequence: " + std::string(seq));
    }

    // Read off the count.
    int read_number = 0;
    for (auto it = seq.begin(); it != type; ++it) {
        read_number = read_number * 10 + static_cast<int>(*it - '0');
    }

    // Drop the op we read and return it.
    seq.remove_prefix(type - seq.begin() + 1);
    return std::make_pair(*type, read_number);
}

}  // namespace

int parse_cigar(std::string_view cigar, dorado::AlignmentResult& result) {
    bool first = true;
    result.strand_start = 0;
    result.num_insertions = 0;
    result.num_deletions = 0;
    result.num_aligned = 0;
    char type;
    int length;
    int hard_clipped = 0;
    while (!cigar.empty()) {
        std::tie(type, length) = next_op(cigar);
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

std::vector<AlignmentResult> parse_sam_lines(std::string_view sam_content,
                                             std::string_view query_seq,
                                             std::string_view query_qual) {
    std::vector<AlignmentResult> results;
    utils::NewlineSeparatedStream sam_content_stream(sam_content);
    std::unordered_map<std::string_view, int> reference_length;

    // read header
    while (sam_content_stream.peek().value_or("").starts_with('@')) {
        utils::TabSeparatedStream header_line(sam_content_stream.getline().value());

        // Read the genome sequence lengths from the header
        std::string_view sq, reference, length_field;
        header_line >> sq >> reference >> length_field;
        if (sq == "@SQ") {
            utils::trace_log("{} length_field: {}", __func__, length_field);
            int ref_length =
                    utils::from_chars<int>(length_field.substr(3, length_field.size())).value();
            reference_length[reference.substr(3, reference.size())] = ref_length;
        }
    }

    // Read every alignment from the SAM file
    for (std::string_view sam_line; sam_content_stream >> sam_line;) {
        AlignmentResult res{};

        // required fields
        std::string_view seq_name, cigar, rnext, aligned_seq, aligned_qstring;
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

        res.num_events = res.sequence == "*" ? 0 : static_cast<int>(res.sequence.size());
        res.mapping_quality = map_quality;

        struct TypeValuePair {
            std::string_view Type, Value;
        };

        std::map<std::string_view, TypeValuePair> opt_values;

        if (res.genome != "*") {
            // optional fields
            for (std::string_view field; sam_line_istream >> field;) {
                if (field.length() < 5 || field[2] != ':' || field[4] != ':') {
                    throw std::runtime_error("optional SAM field '" + std::string(field) +
                                             "' could not be parsed.");
                }
                std::string_view key = field.substr(0, 2);
                std::string_view type = field.substr(3, 1);
                std::string_view value = field.substr(5);
                opt_values[key] = TypeValuePair{type, value};
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
                auto opt_NM = opt_values.find("NM");
                if (opt_NM == opt_values.end()) {
                    throw std::runtime_error("Input SAM line for read ID " + std::string(seq_name) +
                                             " does not contain required 'NM' tag");
                }
                utils::trace_log("{} opt_values.at(\"NM\").Value: {}", __func__,
                                 opt_NM->second.Value);
                int edit_distance = utils::from_chars<int>(opt_values.at("NM").Value).value();
                int num_mismatches = edit_distance - res.num_insertions - res.num_deletions;
                res.num_correct = res.num_aligned - num_mismatches;
                res.identity = float(res.num_correct) / float(res.num_aligned);
                res.accuracy = float(res.num_correct) /
                               float(res.num_aligned + res.num_insertions + res.num_deletions);
                auto opt_AS = opt_values.find("AS");
                if (opt_AS != opt_values.end()) {
                    utils::trace_log("{} opt_values.at(\"AS\").Value: {}", __func__,
                                     opt_AS->second.Value);
                    res.strand_score = utils::from_chars<int>(opt_AS->second.Value).value();
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
        results.emplace_back(std::move(res));
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
