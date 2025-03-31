#include "messages.h"

#include "modbase/ModBaseContext.h"
#include "stereo_features.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"
#include "utils/types.h"

#include <htslib/sam.h>

namespace dorado {

bool is_read_message(const Message &message) {
    return std::holds_alternative<SimplexReadPtr>(message) ||
           std::holds_alternative<DuplexReadPtr>(message);
}

uint64_t SimplexRead::get_end_time_ms() const {
    return read_common.start_time_ms +
           ((end_sample - start_sample) * 1000) /
                   read_common.sample_rate;  //TODO get rid of the trimmed thing?
}

std::string ReadCommon::generate_read_group() const {
    std::string read_group;
    if (!run_id.empty()) {
        read_group = run_id + '_';
        if (model_name.empty()) {
            read_group += "unknown";
        } else {
            read_group += model_name;
        }
        if (!barcode.empty() && barcode != UNCLASSIFIED) {
            read_group += '_' + barcode;
        }
    }
    return read_group;
}

void ReadCommon::generate_read_tags(bam1_t *aln, bool emit_moves, bool is_duplex_parent) const {
    float qs = calculate_mean_qscore();
    bam_aux_append(aln, "qs", 'f', sizeof(qs), (uint8_t *)&qs);

    float du = (float)(get_raw_data_samples() + num_trimmed_samples) / (float)sample_rate;
    bam_aux_append(aln, "du", 'f', sizeof(du), (uint8_t *)&du);

    int ns = int(get_raw_data_samples() + num_trimmed_samples);
    bam_aux_append(aln, "ns", 'i', sizeof(ns), (uint8_t *)&ns);

    int ts = int(num_trimmed_samples);
    bam_aux_append(aln, "ts", 'i', sizeof(ts), (uint8_t *)&ts);

    int mx = attributes.mux;
    bam_aux_append(aln, "mx", 'i', sizeof(mx), (uint8_t *)&mx);

    int ch = attributes.channel_number;
    bam_aux_append(aln, "ch", 'i', sizeof(ch), (uint8_t *)&ch);

    bam_aux_append(aln, "st", 'Z', int(attributes.start_time.length() + 1),
                   (uint8_t *)attributes.start_time.c_str());

    if (primer_classification.orientation != StrandOrientation::UNKNOWN) {
        auto sense_data = uint8_t(to_char(primer_classification.orientation));
        bam_aux_append(aln, "TS", 'A', 1, &sense_data);
    }
    if (!primer_classification.umi_tag_sequence.empty()) {
        auto len = int(primer_classification.umi_tag_sequence.size()) + 1;
        auto data = (const uint8_t *)primer_classification.umi_tag_sequence.c_str();
        bam_aux_append(aln, "RX", 'Z', len, data);
    }

    // For reads which are the result of read splitting, the read number will be set to -1
    int rn = attributes.read_number;
    bam_aux_append(aln, "rn", 'i', sizeof(rn), (uint8_t *)&rn);

    bam_aux_append(aln, "fn", 'Z', int(attributes.filename.length() + 1),
                   (uint8_t *)attributes.filename.c_str());

    float sm = shift;
    bam_aux_append(aln, "sm", 'f', sizeof(sm), (uint8_t *)&sm);

    float sd = scale;
    bam_aux_append(aln, "sd", 'f', sizeof(sd), (uint8_t *)&sd);

    bam_aux_append(aln, "sv", 'Z', int(scaling_method.size() + 1),
                   (uint8_t *)scaling_method.c_str());

    int32_t dx = (is_duplex_parent ? -1 : 0);
    bam_aux_append(aln, "dx", 'i', sizeof(dx), (uint8_t *)&dx);

    auto rg = generate_read_group();
    if (!rg.empty()) {
        bam_aux_append(aln, "RG", 'Z', int(rg.length() + 1), (uint8_t *)rg.c_str());
    }

    if (!parent_read_id.empty()) {
        bam_aux_append(aln, "pi", 'Z', int(parent_read_id.size() + 1),
                       (uint8_t *)parent_read_id.c_str());
        // For split reads, also store the start coordinate of the new read
        // in the original signal.
        bam_aux_append(aln, "sp", 'i', sizeof(split_point), (uint8_t *)&split_point);
    }

    if (emit_moves) {
        std::vector<uint8_t> m(moves.size() + 1, 0);
        m[0] = uint8_t(model_stride);

        for (size_t idx = 0; idx < moves.size(); idx++) {
            m[idx + 1] = static_cast<uint8_t>(moves[idx]);
        }

        bam_aux_update_array(aln, "mv", 'c', int(m.size()), (uint8_t *)m.data());
    }

    if (rna_poly_tail_length != ReadCommon::POLY_TAIL_NOT_ENABLED) {
        bam_aux_append(aln, "pt", 'i', sizeof(rna_poly_tail_length),
                       (uint8_t *)&rna_poly_tail_length);
    }
}

void ReadCommon::generate_duplex_read_tags(bam1_t *aln) const {
    float qs = calculate_mean_qscore();
    bam_aux_append(aln, "qs", 'f', sizeof(qs), (uint8_t *)&qs);
    uint32_t duplex = 1;
    bam_aux_append(aln, "dx", 'i', sizeof(duplex), (uint8_t *)&duplex);

    int mx = attributes.mux;
    bam_aux_append(aln, "mx", 'i', sizeof(mx), (uint8_t *)&mx);

    int ch = attributes.channel_number;
    bam_aux_append(aln, "ch", 'i', sizeof(ch), (uint8_t *)&ch);

    bam_aux_append(aln, "st", 'Z', int(attributes.start_time.length() + 1),
                   (uint8_t *)attributes.start_time.c_str());

    auto rg = generate_read_group();
    if (!rg.empty()) {
        bam_aux_append(aln, "RG", 'Z', int(rg.length() + 1), (uint8_t *)rg.c_str());
    }

    if (!parent_read_id.empty()) {
        bam_aux_append(aln, "pi", 'Z', int(parent_read_id.size() + 1),
                       (uint8_t *)parent_read_id.c_str());
    }
}

void ReadCommon::generate_modbase_tags(bam1_t *aln, std::optional<uint8_t> threshold) const {
    if (!mod_base_info) {
        return;
    }
    if (!threshold.has_value()) {
        throw std::logic_error("Cannot generate modbase tags without a valid threshold.");
    }

    const size_t num_channels = mod_base_info->alphabet.size();
    const std::string cardinal_bases = "ACGT";
    char current_cardinal = 0;
    if (seq.length() * num_channels != base_mod_probs.size()) {
        throw std::runtime_error(
                "Mismatch between base_mod_probs size and sequence length * num channels in "
                "modbase_alphabet!");
    }

    std::string modbase_string = "";
    std::vector<uint8_t> modbase_prob;

    // Create a mask indicating which bases are modified.
    std::unordered_map<char, bool> base_has_context = {
            {'A', false}, {'C', false}, {'G', false}, {'T', false}};
    modbase::ModBaseContext context_handler;
    if (!mod_base_info->context.empty()) {
        if (!context_handler.decode(mod_base_info->context)) {
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
    context_handler.update_mask(modbase_mask, seq, mod_base_info->alphabet, base_mod_probs,
                                *threshold);

    if (is_duplex) {
        // If this is a duplex read, we need to compute the reverse complement mask and combine it
        auto reverse_complemented_seq = utils::reverse_complement(seq);

        // Compute the reverse complement mask
        auto modbase_mask_rc = context_handler.get_sequence_mask(reverse_complemented_seq);

        auto reverseMatrix = [](const std::vector<uint8_t> &matrix, int m_num_states) {
            int numRows = static_cast<int>(matrix.size()) / static_cast<int>(m_num_states);
            std::vector<uint8_t> reversedMatrix(matrix.size());

            for (int i = 0; i < numRows; ++i) {
                for (int j = 0; j < m_num_states; ++j) {
                    reversedMatrix[i * m_num_states + j] =
                            matrix[(numRows - 1 - i) * m_num_states + j];
                }
            }

            return reversedMatrix;
        };

        int num_states = static_cast<int>(base_mod_probs.size()) / static_cast<int>(seq.size());
        // Update the context mask using the reversed sequence
        context_handler.update_mask(modbase_mask_rc, reverse_complemented_seq,
                                    mod_base_info->alphabet,
                                    reverseMatrix(base_mod_probs, num_states), *threshold);

        // Reverse the mask in-place
        std::reverse(modbase_mask_rc.begin(), modbase_mask_rc.end());

        // Combine the original and reverse complement masks
        // Using std::transform for better readability and potential efficiency
        std::transform(modbase_mask.begin(), modbase_mask.end(), modbase_mask_rc.begin(),
                       modbase_mask.begin(), std::plus<>());
    }

    // Iterate over the provided alphabet and find all the channels we need to write out
    for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
        if (cardinal_bases.find(mod_base_info->alphabet[channel_idx]) != std::string::npos) {
            // A cardinal base
            current_cardinal = mod_base_info->alphabet[channel_idx][0];
        } else {
            // A modification on the previous cardinal base
            std::string bam_name = mod_base_info->alphabet[channel_idx];
            if (!utils::validate_bam_tag_code(bam_name)) {
                return;
            }

            // Write out the results we found
            modbase_string += std::string(1, current_cardinal) + "+" + bam_name;
            modbase_string += base_has_context[current_cardinal] ? "?" : ".";
            int skipped_bases = 0;
            for (size_t base_idx = 0; base_idx < seq.size(); base_idx++) {
                if (seq[base_idx] == current_cardinal) {
                    if (modbase_mask[base_idx]) {
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

    if (is_duplex) {
        // Having done the strand in the forward direction, if the read is duplex we need to also process its complement
        // There is some code repetition here, but it makes it more readable.
        for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
            if (cardinal_bases.find(mod_base_info->alphabet[channel_idx]) != std::string::npos) {
                // A cardinal base
                current_cardinal = mod_base_info->alphabet[channel_idx][0];
            } else {
                auto cardinal_complement = utils::complement_table[current_cardinal];
                // A modification on the previous cardinal base
                std::string bam_name = mod_base_info->alphabet[channel_idx];
                if (!utils::validate_bam_tag_code(bam_name)) {
                    return;
                }

                modbase_string += std::string(1, cardinal_complement) + "-" + bam_name;
                modbase_string += base_has_context[current_cardinal] ? "?" : ".";
                int skipped_bases = 0;
                for (size_t base_idx = 0; base_idx < seq.size(); base_idx++) {
                    if (seq[base_idx] == cardinal_complement) {  // complement
                        if (modbase_mask[base_idx]) {            // Not sure this one is right
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
    }

    int seq_len = int(seq.length());
    bam_aux_append(aln, "MN", 'i', sizeof(seq_len), (uint8_t *)&seq_len);
    bam_aux_append(aln, "MM", 'Z', int(modbase_string.length() + 1),
                   (uint8_t *)modbase_string.c_str());
    bam_aux_update_array(aln, "ML", 'C', int(modbase_prob.size()), (uint8_t *)modbase_prob.data());
}

float ReadCommon::calculate_mean_qscore() const {
    if (is_rna_model) {
        const size_t polya_start = utils::find_rna_polya(seq);
        spdlog::trace("calculate_mean_qscore rna - len:{} polya_start_idx: {}, polya_trim_len:{}",
                      seq.size(), polya_start, seq.size() - polya_start);
        if (polya_start == 0) {
            return utils::mean_qscore_from_qstring(qstring);
        }
        return utils::mean_qscore_from_qstring(std::string_view{qstring}.substr(0, polya_start));
    }

    // If Q-score start position is greater than the
    // read length, then calculate mean Q-score from the
    // start of the read.
    if (qstring.length() <= mean_qscore_start_pos) {
        return utils::mean_qscore_from_qstring(qstring);
    }
    return utils::mean_qscore_from_qstring(std::string_view{qstring}.substr(mean_qscore_start_pos));
}

std::vector<BamPtr> ReadCommon::extract_sam_lines(bool emit_moves,
                                                  std::optional<uint8_t> modbase_threshold,
                                                  bool is_duplex_parent) const {
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

    bam1_t *aln = bam_init1();
    uint32_t flags = 4;     // 4 = UNMAPPED
    int leftmost_pos = -1;  // UNMAPPED - will be written as 0
    int map_q = 0;          // UNMAPPED
    int next_pos = -1;      // UNMAPPED - will be written as 0

    // Convert string qscore to phred vector.
    std::vector<uint8_t> qscore;
    std::transform(qstring.begin(), qstring.end(), std::back_inserter(qscore),
                   [](char c) { return static_cast<uint8_t>(c - 33); });

    bam_set1(aln, read_id.length(), read_id.c_str(), uint16_t(flags), -1, leftmost_pos,
             uint8_t(map_q), 0, nullptr, -1, next_pos, 0, seq.length(), seq.c_str(),
             (char *)qscore.data(), 0);

    if (!barcode.empty() && barcode != UNCLASSIFIED) {
        bam_aux_append(aln, "BC", 'Z', int(barcode.length() + 1), (uint8_t *)barcode.c_str());
    }

    if (is_duplex) {
        generate_duplex_read_tags(aln);
    } else {
        generate_read_tags(aln, emit_moves, is_duplex_parent);
    }
    generate_modbase_tags(aln, modbase_threshold);
    alns.push_back(BamPtr(aln));

    return alns;
}

ReadCommon &get_read_common_data(Message &message) {
    return const_cast<ReadCommon &>(get_read_common_data(const_cast<const Message &>(message)));
}

const ReadCommon &get_read_common_data(const Message &message) {
    if (!is_read_message(message)) {
        throw std::invalid_argument("Message is not a read");
    } else {
        if (std::holds_alternative<SimplexReadPtr>(message)) {
            return std::get<SimplexReadPtr>(message)->read_common;
        } else {
            return std::get<DuplexReadPtr>(message)->read_common;
        }
    }
}

void materialise_read_raw_data(Message &message) {
    if (std::holds_alternative<DuplexReadPtr>(message)) {
        // Note: we could deallocate stereo_feature_inputs fields,
        // but this made a negligible difference to overall memory usage.
        auto &duplex_read = *std::get<DuplexReadPtr>(message);
        duplex_read.read_common.raw_data =
                generate_stereo_features(duplex_read.stereo_feature_inputs);
    }
}

ReadPair::ReadData ReadPair::ReadData::from_read(const SimplexRead &read,
                                                 uint64_t seq_start,
                                                 uint64_t seq_end) {
    ReadData data;
    data.read_common = read.read_common;
    data.seq_start = seq_start;
    data.seq_end = seq_end;
    return data;
}

}  // namespace dorado