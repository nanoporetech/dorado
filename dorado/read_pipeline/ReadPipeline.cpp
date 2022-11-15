#include "ReadPipeline.h"

#include "utils/base_mod_utils.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <iomanip>
#include <sstream>

using namespace std::chrono_literals;

namespace {
bool get_modbase_channel_name(std::string& channel_name, const std::string& mod_abbreviation) {
    static const std::map<std::string, std::string> modbase_name_map = {// A
                                                                        {"6mA", "a"},
                                                                        // C
                                                                        {"5mC", "m"},
                                                                        {"5hmC", "h"},
                                                                        {"5fc", "f"},
                                                                        {"5caC", "c"},
                                                                        // G
                                                                        {"8oxoG", "o"},
                                                                        // T
                                                                        {"5hmU", "g"},
                                                                        {"5fU", "e"},
                                                                        {"5caU", "b"}};

    if (modbase_name_map.find(mod_abbreviation) != modbase_name_map.end()) {
        channel_name = modbase_name_map.at(mod_abbreviation);
        return true;
    }

    // Check the supplied mod abbreviation is a simple integer and if so, assume it's a CHEBI code.
    if (mod_abbreviation.find_first_not_of("0123456789") == std::string::npos) {
        channel_name = mod_abbreviation;
        return true;
    }

    spdlog::error("Unknown modified base abbreviation: {}", mod_abbreviation);
    return false;
}
}  // namespace

namespace dorado {

std::vector<std::string> Read::generate_read_tags(bool emit_moves) const {
    // GCC doesn't support <format> yet...

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << shift;
    std::string shift_str = stream.str();

    stream = std::stringstream();
    stream << std::fixed << std::setprecision(3) << scale;
    std::string scale_str = stream.str();

    std::vector<std::string> tags = {
            "qs:i:" + std::to_string(static_cast<int>(
                              std::round(utils::mean_qscore_from_qstring(qstring)))),
            "du:f:" + std::to_string((raw_data.size(0) + num_trimmed_samples) / sample_rate),
            "ns:i:" + std::to_string(raw_data.size(0) + num_trimmed_samples),
            "ts:i:" + std::to_string(num_trimmed_samples),
            "mx:i:" + std::to_string(attributes.mux),
            "ch:i:" + std::to_string(attributes.channel_number),
            "st:Z:" + attributes.start_time,
            "rn:i:" + std::to_string(attributes.read_number),
            "f5:Z:" + attributes.fast5_filename,
            "sm:f:" + shift_str,
            "sd:f:" + scale_str,
            "sv:Z:quantile"};

    if (emit_moves) {
        const std::string tag{"mv:B:c," + std::to_string(model_stride)};
        std::string movess(moves.size() * 2 + tag.size(), ',');

        for (size_t idx = 0; idx < tag.size(); idx++) {
            movess[idx] = tag[idx];
        }

        for (size_t idx = 0; idx < moves.size(); idx++) {
            movess[idx * 2 + tag.size() + 1] = static_cast<char>(moves[idx] + 48);
        }

        tags.push_back(movess);
    }

    return tags;
}

std::vector<std::string> Read::generate_duplex_read_tags() const {
    std::vector<std::string> tags = {"qs:i:" + std::to_string(static_cast<int>(std::round(
                                                       utils::mean_qscore_from_qstring(qstring))))};
    return tags;
}

std::vector<std::string> Read::extract_sam_lines(bool emit_moves, bool duplex) const {
    if (read_id.empty()) {
        throw std::runtime_error("Empty read_name string provided");
    }
    if (seq.size() != qstring.size()) {
        throw std::runtime_error("Sequence and qscore do not match size for read id " + read_id);
    }
    if (seq.empty()) {
        throw std::runtime_error("Empty sequence and qstring provided for read id " + read_id);
    }

    std::ostringstream read_tags_stream;
    std::vector<std::string> read_tags;

    if (duplex) {
        read_tags = generate_duplex_read_tags();
    } else {
        read_tags = generate_read_tags(emit_moves);
    }

    for (const auto& tag : read_tags) {
        read_tags_stream << "\t" << tag;
    }

    std::vector<std::string> sam_lines;
    if (mappings.empty()) {
        uint32_t flags = 4;              // 4 = UNMAPPED
        std::string ref_seq = "*";       // UNMAPPED
        int leftmost_pos = -1;           // UNMAPPED - will be written as 0
        int map_q = 0;                   // UNMAPPED
        std::string cigar_string = "*";  // UNMAPPED
        std::string r_next = "*";
        int next_pos = -1;  // UNMAPPED - will be written as 0
        size_t template_length = seq.size();

        std::ostringstream sam_line;
        sam_line << read_id << "\t"             // QNAME
                 << flags << "\t"               // FLAG
                 << ref_seq << "\t"             // RNAME
                 << (leftmost_pos + 1) << "\t"  // POS
                 << map_q << "\t"               // MAPQ
                 << cigar_string << "\t"        // CIGAR
                 << r_next << "\t"              // RNEXT
                 << (next_pos + 1) << "\t"      // PNEXT
                 << (template_length) << "\t"   // TLEN
                 << seq << "\t"                 // SEQ
                 << qstring;                    // QUAL

        sam_line << read_tags_stream.str();
        sam_lines.push_back(sam_line.str());
    }

    for (const auto& mapping : mappings) {
        throw std::runtime_error("Mapped alignments not yet implemented");
    }

    auto mod_base_string = generate_modbase_string();
    if (!mod_base_string.empty()) {
        sam_lines.front() += "\t";
        sam_lines.front() += mod_base_string;
    }
    return sam_lines;
}

std::string Read::generate_modbase_string(uint8_t threshold) const {
    if (!base_mod_info) {
        return {};
    }

    const size_t num_channels = base_mod_info->alphabet.size();
    const std::string cardinal_bases = "ACGT";
    char current_cardinal = 0;
    if (seq.length() * num_channels != base_mod_probs.size()) {
        throw std::runtime_error(
                "Mismatch between base_mod_probs size and sequence length * num channels in "
                "modbase_alphabet!");
    }

    std::istringstream mod_name_stream(base_mod_info->long_names);
    std::string modbase_string = "MM:Z:";
    std::string modbase_prob_string = "ML:B:C";

    // Create a mask indicating which bases are modified.
    std::map<char, bool> base_has_context = {
            {'A', false}, {'C', false}, {'G', false}, {'T', false}};
    utils::BaseModContext context_handler;
    if (!base_mod_info->context.empty()) {
        if (!context_handler.decode(base_mod_info->context)) {
            throw std::runtime_error("Invalid base modification context string.");
        }
        for (auto base : cardinal_bases) {
            if (context_handler.motif(base).size() > 1) {
                // If the context is just the single base, then this is equivalent to no context.
                base_has_context[base] = true;
            }
        }
    }
    auto modbase_mask = context_handler.get_sequence_mask(seq);
    context_handler.update_mask(modbase_mask, seq, base_mod_info->alphabet, base_mod_probs,
                                threshold);

    // Iterate over the provided alphabet and find all the channels we need to write out
    for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        if (cardinal_bases.find(base_mod_info->alphabet[channel_idx]) != std::string::npos) {
            // A cardinal base
            current_cardinal = base_mod_info->alphabet[channel_idx];
        } else {
            // A modification on the previous cardinal base
            std::string modbase_name;
            mod_name_stream >> modbase_name;
            std::string bam_name;
            if (!get_modbase_channel_name(bam_name, modbase_name)) {
                return {};
            }

            // Write out the results we found
            modbase_string += std::string(1, current_cardinal) + "+" + bam_name;
            if (base_has_context[current_cardinal]) {
                modbase_string += "?";
            }
            int skipped_bases = 0;
            for (size_t base_idx = 0; base_idx < seq.size(); base_idx++) {
                if (seq[base_idx] == current_cardinal) {
                    if (modbase_mask[base_idx] == 1) {
                        modbase_string += "," + std::to_string(skipped_bases);
                        skipped_bases = 0;
                        modbase_prob_string +=
                                "," +
                                std::to_string(
                                        base_mod_probs[base_idx * num_channels + channel_idx]);
                    } else {
                        // Skip this base
                        skipped_bases++;
                    }
                }
            }
            modbase_string += ";";
        }
    }

    modbase_string += "\t" + modbase_prob_string;
    return modbase_string;
}

void ReadSink::push_read(std::shared_ptr<Read>& read) {
    std::unique_lock<std::mutex> push_read_cv_lock(m_push_read_cv_mutex);
    while (!m_push_read_cv.wait_for(push_read_cv_lock, 100ms,
                                    [this] { return m_reads.size() < m_max_reads; })) {
    }
    {
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_reads.push_back(read);
    }
    m_cv.notify_one();
}

ReadSink::ReadSink(size_t max_reads) : m_max_reads(max_reads) {}

}  // namespace dorado
