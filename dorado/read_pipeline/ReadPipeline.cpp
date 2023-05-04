#include "ReadPipeline.h"

#include "htslib/sam.h"
#include "utils/base_mod_utils.h"
#include "utils/sequence_utils.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <iomanip>
#include <sstream>

using namespace std::chrono_literals;

namespace {
bool get_modbase_channel_name(std::string &channel_name, const std::string &mod_abbreviation) {
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

void Read::generate_read_tags(bam1_t *aln, bool emit_moves) const {
    int qs = static_cast<int>(std::round(utils::mean_qscore_from_qstring(qstring)));
    bam_aux_append(aln, "qs", 'i', sizeof(qs), (uint8_t *)&qs);

    float du = (raw_data.size(0) + num_trimmed_samples) / sample_rate;
    bam_aux_append(aln, "du", 'f', sizeof(du), (uint8_t *)&du);

    int ns = raw_data.size(0) + num_trimmed_samples;
    bam_aux_append(aln, "ns", 'i', sizeof(ns), (uint8_t *)&ns);

    int ts = num_trimmed_samples;
    bam_aux_append(aln, "ts", 'i', sizeof(ts), (uint8_t *)&ts);

    int mx = attributes.mux;
    bam_aux_append(aln, "mx", 'i', sizeof(mx), (uint8_t *)&mx);

    int ch = attributes.channel_number;
    bam_aux_append(aln, "ch", 'i', sizeof(ch), (uint8_t *)&ch);

    bam_aux_append(aln, "st", 'Z', attributes.start_time.length() + 1,
                   (uint8_t *)attributes.start_time.c_str());

    int rn = attributes.read_number;
    bam_aux_append(aln, "rn", 'i', sizeof(rn), (uint8_t *)&rn);

    bam_aux_append(aln, "fn", 'Z', attributes.fast5_filename.length() + 1,
                   (uint8_t *)attributes.fast5_filename.c_str());

    float sm = shift;
    bam_aux_append(aln, "sm", 'f', sizeof(sm), (uint8_t *)&sm);

    float sd = scale;
    bam_aux_append(aln, "sd", 'f', sizeof(sd), (uint8_t *)&sd);

    bam_aux_append(aln, "sv", 'Z', 9, (uint8_t *)"quantile");

    if (run_id != "" && model_name != "") {
        std::string rg(run_id + "_" + model_name);
        bam_aux_append(aln, "RG", 'Z', rg.length() + 1, (uint8_t *)rg.c_str());
    }

    if (emit_moves) {
        std::vector<uint8_t> m(moves.size() + 1, 0);
        m[0] = model_stride;

        for (size_t idx = 0; idx < moves.size(); idx++) {
            m[idx + 1] = static_cast<uint8_t>(moves[idx]);
        }

        bam_aux_update_array(aln, "mv", 'c', m.size(), (uint8_t *)m.data());
    }
}

void Read::generate_duplex_read_tags(bam1_t *aln) const {
    int qs = static_cast<int>(std::round(utils::mean_qscore_from_qstring(qstring)));
    bam_aux_append(aln, "qs", 'i', sizeof(qs), (uint8_t *)&qs);
}

std::vector<BamPtr> Read::extract_sam_lines(bool emit_moves,
                                            bool duplex,
                                            uint8_t modbase_threshold) const {
    if (read_id.empty()) {
        throw std::runtime_error("Empty read_name string provided");
    }
    if (seq.size() != qstring.size()) {
        throw std::runtime_error("Sequence and qscore do not match size for read id " + read_id);
    }
    if (seq.empty()) {
        throw std::runtime_error("Empty sequence and qstring provided for read id " + read_id);
    }

    std::vector<BamPtr> alns;
    if (mappings.empty()) {
        bam1_t *aln = bam_init1();
        uint32_t flags = 4;              // 4 = UNMAPPED
        std::string ref_seq = "*";       // UNMAPPED
        int leftmost_pos = -1;           // UNMAPPED - will be written as 0
        int map_q = 0;                   // UNMAPPED
        std::string cigar_string = "*";  // UNMAPPED
        std::string r_next = "*";
        int next_pos = -1;  // UNMAPPED - will be written as 0
        size_t template_length = seq.size();

        // Convert string qscore to phred vector.
        std::vector<uint8_t> qscore;
        std::transform(qstring.begin(), qstring.end(), std::back_inserter(qscore),
                       [](char c) { return (uint8_t)(c)-33; });

        bam_set1(aln, read_id.length(), read_id.c_str(), flags, -1, leftmost_pos, map_q, 0, nullptr,
                 -1, next_pos, 0, seq.length(), seq.c_str(), (char *)qscore.data(), 0);

        if (duplex) {
            generate_duplex_read_tags(aln);
        } else {
            generate_read_tags(aln, emit_moves);
        }
        generate_modbase_string(aln, modbase_threshold);
        alns.push_back(BamPtr(aln));
    }

    for (const auto &mapping : mappings) {
        throw std::runtime_error("Mapped alignments not yet implemented");
    }

    return alns;
}

void Read::generate_modbase_string(bam1_t *aln, uint8_t threshold) const {
    if (!base_mod_info) {
        return;
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
    std::string modbase_string = "";
    std::vector<uint8_t> modbase_prob;

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
                return;
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
                        modbase_prob.push_back(
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

    bam_aux_append(aln, "MM", 'Z', modbase_string.length() + 1, (uint8_t *)modbase_string.c_str());
    bam_aux_update_array(aln, "ML", 'C', modbase_prob.size(), (uint8_t *)modbase_prob.data());
}

void MessageSink::push_message(Message &&message) {
    const bool success = m_work_queue.try_push(std::move(message));
    // try_push will fail if the sink has been told to terminate.
    // We do not expect to be pushing reads from this source if that is the case.
    assert(success);
}

MessageSink::MessageSink(size_t max_messages) : m_work_queue(max_messages) {}

}  // namespace dorado
