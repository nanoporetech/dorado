#include "beam_search.h"

#include "fast_hash.h"

#include <math.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

namespace {

// 16 bit state supports 7-mers with 4 bases.
typedef int16_t state_t;

const int num_bases = 4;

// This is the data we need to retain for the whole beam
struct BeamElement {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

// This is the data we need to retain for only the previous timestep (block) in the beam
//  (and what we construct for the new timestep)
struct BeamFrontElement {
    uint64_t hash;
    float score;
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

float log_sum_exp(float x, float y, float t) {
    float abs_diff = fabsf(x - y) / t;
    return fmaxf(x, y) + ((abs_diff < 17.0f) ? (log1pf(expf(-abs_diff)) * t) : 0.0f);
}

bool score_sort(const BeamFrontElement& a, const BeamFrontElement& b) { return a.score > b.score; }

int get_num_states(size_t num_trans_states) {
#ifdef REMOVE_FIXED_BEAM_STAYS
    if (num_trans_states % num_bases != 0) {
        throw std::runtime_error("Unexpected number of transition states in beam search decode.");
    }
    return int(num_trans_states / num_bases);
#else
    if (num_trans_states % (num_bases + 1) != 0) {
        throw std::runtime_error("Unexpected number of transition states in beam search decode.");
    }
    return int(num_trans_states / (num_bases + 1));
#endif
}

std::tuple<std::string, std::string> generate_sequence(const std::vector<uint8_t>& moves,
                                                       const std::vector<int32_t>& states,
                                                       const std::vector<float>& qual_data,
                                                       float shift,
                                                       float scale) {
    size_t seqPos = 0;
    size_t num_blocks = moves.size();
    size_t seqLen = accumulate(moves.begin(), moves.end(), 0);

    std::string sequence(seqLen, 'N');
    std::string qstring(seqLen, '!');
    std::vector<char> alphabet = {'A', 'C', 'G', 'T'};
    std::vector<float> baseProbs(seqLen), totalProbs(seqLen);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        int state = states[blk];
        int move = int(moves[blk]);
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = int(seqPos) + offset;

        // Get the probability for the called base.
        baseProbs[probPos] += qual_data[blk * alphabet.size() + base];

        // Accumulate the total probability for all possible bases at this position, for normalization.
        for (size_t k = 0; k < alphabet.size(); ++k) {
            totalProbs[probPos] += qual_data[blk * alphabet.size() + k];
        }

        if (blk == 0) {
            sequence[seqPos++] = char(base);
        } else {
            for (int j = 0; j < move; ++j) {
                sequence[seqPos++] = char(base);
            }
        }
    }

    for (size_t i = 0; i < seqLen; ++i) {
        sequence[i] = alphabet[int(sequence[i])];
        baseProbs[i] = 1.0f - (baseProbs[i] / totalProbs[i]);
        baseProbs[i] = -10.0f * log10f(baseProbs[i]);
        float qscore = baseProbs[i] * scale + shift;
        qscore = std::min(90.0f, qscore);
        qscore = std::max(1.0f, qscore);
        qstring[i] = char(33.5f + qscore);
    }

    return make_tuple(sequence, qstring);
}

}  // anonymous namespace

template <typename T>
float beam_search(const T* scores,
                  size_t scores_block_stride,
                  const T* back_guide,
                  const float* posts,
                  size_t num_states,
                  size_t num_blocks,
                  size_t max_beam_width,
                  float beam_cut,
                  float fixed_stay_score,
                  std::vector<int32_t>& states,
                  std::vector<uint8_t>& moves,
                  std::vector<float>& qual_data,
                  float temperature) {
    if (max_beam_width > 256) {
        throw std::range_error("Beamsearch max_beam_width cannot be greater than 256.");
    }

    // Some values we need
#ifdef REMOVE_FIXED_BEAM_STAYS
    size_t num_transitions = num_states * num_bases;
#else
    size_t num_transitions = num_states * (num_bases + 1);
#endif
    constexpr uint64_t hash_seed = 0x880355f21e6d1965ULL;
    const float log_beam_cut =
            (beam_cut > 0.0f) ? (temperature * logf(beam_cut)) : std::numeric_limits<float>::max();

    // Create the beam.  We need to keep beam_width elements for each block, plus the initial state
    std::vector<BeamElement> beam_vector(max_beam_width * (num_blocks + 1));

    // Create the previous and current beam fronts
    // Each existing element can be extended by one of num_bases, or be a stay.
    size_t max_beam_candidates = (num_bases + 1) * max_beam_width;

    std::vector<BeamFrontElement> beam_front_vector_1(max_beam_candidates);
    std::vector<BeamFrontElement> beam_front_vector_2(max_beam_candidates);
    std::vector<BeamFrontElement>* current_beam_front = &beam_front_vector_1;
    std::vector<BeamFrontElement>* prev_beam_front = &beam_front_vector_2;

    // Find the score an initial element needs in order to make it into the beam
    T beam_init_threshold = std::numeric_limits<T>::lowest();
    if (max_beam_width < num_states) {
        // Copy the first set of back guides and sort to extract max_beam_width highest elements
        std::vector<T> sorted_back_guides(num_states);
        memcpy(sorted_back_guides.data(), back_guide, num_states * sizeof(T));

        // Note we don't need a full sort here to get the max_beam_width highest values
        std::nth_element(sorted_back_guides.begin(),
                         sorted_back_guides.begin() + max_beam_width - 1, sorted_back_guides.end(),
                         std::greater<T>());
        beam_init_threshold = sorted_back_guides[max_beam_width - 1];
    }

    // Initialise the beam
    for (size_t state = 0, beam_element = 0; state < num_states && beam_element < max_beam_width;
         state++) {
        if (back_guide[state] >= beam_init_threshold) {
            // Note that this first element has a prev_element_index of 0
            (*prev_beam_front)[beam_element++] = {chainfasthash64(hash_seed, state), 0.0f,
                                                  state_t(state), 0, false};
        }
    }

    // Copy this initial beam front into the beam persistent state
    size_t current_beam_width = std::min(max_beam_width, num_states);
    for (size_t element_idx = 0; element_idx < current_beam_width; element_idx++) {
        beam_vector[element_idx].state = (*prev_beam_front)[element_idx].state;
        beam_vector[element_idx].prev_element_index =
                (*prev_beam_front)[element_idx].prev_element_index;
        beam_vector[element_idx].stay = (*prev_beam_front)[element_idx].stay;
    }

    // Iterate through blocks, extending beam
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        const T* block_scores = scores + (block_idx * scores_block_stride);
        const T* block_back_scores = back_guide + ((block_idx + 1) * num_states);
#ifdef REMOVE_FIXED_BEAM_STAYS
        /*  kmer transitions order:
	 *  N^K , N array
	 *  Elements stored as resulting kmer and modifying action (stays have a fixed score and are not computed).
	 *  Kmer index is lexographic with most recent base in the fastest index
	 *
	 *  E.g.  AGT has index (4^2, 4, 1) . (0, 2, 3) == 11
	 *  The modifying action is
	 *    0: Remove A from beginning
	 *    1: Remove C from beginning
	 *    2: Remove G from beginning
	 *    3: Remove T from beginning
	 *
	 *  Transition (movement) ACGTT (111) -> CGTTG (446) has index 446 * 4 + 0 = 1784
	 */
        auto generate_move_index = [](state_t previous_state, state_t new_state, size_t num_bases,
                                      size_t num_states) {
            return state_t(new_state * num_bases + ((previous_state * num_bases) / num_states));
        };
#else   // REMOVE_FIXED_BEAM_STAYS
        /*  kmer transitions order:
         *  N^K , (N + 1) array
         *  Elements stored as resulting kmer and modifying action (0 == stay).
         *  Kmer index is lexographic with most recent base in the fastest index
         *
         *  E.g.  AGT has index (4^2, 4, 1) . (0, 2, 3) == 11
         *  The modifying action is
         *    0: stay
         *    1: Remove A from beginning
         *    2: Remove C from beginning
         *    3: Remove G from beginning
         *    4: Remove T from beginning
         *
         *  Transition (movement) ACGTT (111) -> CGTTG (446) has index 446 * 5 + 1 = 2231
         *  Transition (stay) ACGTT (111) -> ACGTT (111) has index 111 * 5 + 0 = 555
         */

        auto generate_move_index = [](state_t previous_state, state_t new_state, size_t num_bases,
                                      size_t num_states) {
            return state_t(new_state * (num_bases + 1) +
                           (1 + (previous_state * num_bases) / num_states));
        };
        auto generate_stay_index = [](state_t state, size_t num_bases) {
            return state_t(state * (num_bases + 1));
        };
#endif  // REMOVE_FIXED_BEAM_STAYS \
        // Generate list of candidate elements for this timestep (block)
        size_t new_elem_count = 0;
        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; prev_elem_idx++) {
            const auto& previous_element = (*prev_beam_front)[prev_elem_idx];

            // Expand all the possible steps
            for (size_t new_base = 0; new_base < num_bases; new_base++) {
                state_t new_state =
                        state_t((previous_element.state * num_bases) % num_states + new_base);
                const state_t move_idx = generate_move_index(previous_element.state, new_state,
                                                             num_bases, num_states);
                float new_score = previous_element.score + float(block_scores[move_idx]) +
                                  float(block_back_scores[new_state]);
                uint64_t new_hash = chainfasthash64(previous_element.hash, new_state);

                // Add new element to the candidate list
                (*current_beam_front)[new_elem_count++] = {new_hash, new_score, new_state,
                                                           (uint8_t)prev_elem_idx, false};
            }
        }

        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; prev_elem_idx++) {
            const auto& previous_element = (*prev_beam_front)[prev_elem_idx];
            // Add the possible stay
#ifdef REMOVE_FIXED_BEAM_STAYS
            const float stay_score = previous_element.score + fixed_stay_score +
                                     float(block_back_scores[previous_element.state]);
#else
            const state_t stay_idx = generate_stay_index(previous_element.state, num_bases);
            const float stay_score = previous_element.score + block_scores[stay_idx] +
                                     float(block_back_scores[previous_element.state]);
#endif
            (*current_beam_front)[new_elem_count++] = {previous_element.hash, stay_score,
                                                       previous_element.state,
                                                       (uint8_t)prev_elem_idx, true};
        }

        // For each new stay, see if any steps result in the same sequence hash, and merge if so
        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; prev_elem_idx++) {
            // The index of the stay in the beamfront
            size_t stay_elem_idx = num_bases * current_beam_width + prev_elem_idx;
            // latest base is in smallest bits
            int stay_latest_base = int((*current_beam_front)[stay_elem_idx].state % num_bases);

            // Go through all the possible step extensions that match this destination base with the stay and compare
            //  their hashes, merging if we find any
            for (size_t prev_elem_comp_idx = 0; prev_elem_comp_idx < current_beam_width;
                 prev_elem_comp_idx++) {
                size_t step_elem_idx = prev_elem_comp_idx * num_bases + stay_latest_base;
                if ((*current_beam_front)[stay_elem_idx].hash ==
                    (*current_beam_front)[step_elem_idx].hash) {
                    if ((*current_beam_front)[stay_elem_idx].score >
                        (*current_beam_front)[step_elem_idx].score) {
                        // Fold the step into the stay
                        (*current_beam_front)[stay_elem_idx].score = log_sum_exp(
                                (*current_beam_front)[stay_elem_idx].score,
                                (*current_beam_front)[step_elem_idx].score, temperature);
                        // The step element will end up last, sorted by score
                        (*current_beam_front)[step_elem_idx].score =
                                -std::numeric_limits<float>::max();
                    } else {
                        // Fold the stay into the step
                        (*current_beam_front)[step_elem_idx].score = log_sum_exp(
                                (*current_beam_front)[stay_elem_idx].score,
                                (*current_beam_front)[step_elem_idx].score, temperature);
                        // The stay element will end up last, sorted by score
                        (*current_beam_front)[stay_elem_idx].score =
                                -std::numeric_limits<float>::max();
                    }
                }
            }
        }

        // There are now `new_elem_count` elements in the list.  Let's get the max
        float max_score = -std::numeric_limits<float>::max();
        for (size_t elem_idx = 0; elem_idx < new_elem_count; elem_idx++) {
            if ((*current_beam_front)[elem_idx].score > max_score)
                max_score = (*current_beam_front)[elem_idx].score;
        }

        // Starting point for finding the cutoff score is the beam cut score
        float beam_cutoff_score = max_score - log_beam_cut;

        auto get_elem_count = [](const std::vector<BeamFrontElement>* current_beam_front,
                                 size_t new_elem_count, float beam_score) {
            // Count the elements which meet the beam score
            size_t count = 0;
            for (size_t elem_idx = 0; elem_idx < new_elem_count; elem_idx++) {
                if ((*current_beam_front)[elem_idx].score >= beam_score)
                    count++;
            }
            return count;
        };

        // Count the elements which meet the min score
        size_t elem_count = get_elem_count(current_beam_front, new_elem_count, beam_cutoff_score);

        if (elem_count > max_beam_width) {
            // Need to find a score which doesn't return too many scores, but doesn't reduce beam width too much
            size_t min_beam_width =
                    (max_beam_width * 8) / 10;  // 80% of beam width is the minimum we accept.
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            static const int MAX_GUESSES = 10;
            while ((elem_count > max_beam_width || elem_count < min_beam_width) &&
                   num_guesses < MAX_GUESSES) {
                if (elem_count > max_beam_width) {
                    // Make a higher guess
                    low_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;  // binary search.
                } else {
                    // Make a lower guess
                    hi_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;  // binary search.
                }
                elem_count = get_elem_count(current_beam_front, new_elem_count, beam_cutoff_score);
                num_guesses++;
            }
            // If we made 10 guesses and didn't find a suitable score, a couple of things may have happened:
            // 1: we just haven't completed the binary search yet (there is a good score in there somewhere but we didn't find it.)
            //  - in this case we should just pick the higher of the two current search limits to get the top N elements)
            // 2: there is no good score, as max_score returns more than beam_width elements (i.e. more than the whole beam width has max_score)
            //  - in this case we should just take max_beam_width of the top-scoring elements
            // 3: there is no good score as all the elements from <80% of the beam to >100% have the same score.
            //  - in this case we should just take the hi_score and accept it will return us less than 80% of the beam
            if (num_guesses == MAX_GUESSES) {
                beam_cutoff_score = hi_score;
                elem_count = get_elem_count(current_beam_front, new_elem_count, beam_cutoff_score);
            }
        }
        // Clamp the element count to the max beam width in case of failure 2 from above.
        elem_count = std::min(elem_count, max_beam_width);

        size_t write_idx = 0;
        for (unsigned int read_idx = 0; read_idx < new_elem_count; read_idx++) {
            if ((*current_beam_front)[read_idx].score >= beam_cutoff_score) {
                if (write_idx < max_beam_width) {
                    (*prev_beam_front)[write_idx] = (*current_beam_front)[read_idx];
                    write_idx++;
                }
            }
        }

        // At the last timestep, we need to sort the prev_beam_front as the best path needs to be at the start
        // NOTE: We only want the top score out, so the cutoff is set to 1
        if (block_idx == num_blocks - 1) {
            merge_sort(prev_beam_front->data(), elem_count, 1, score_sort);
        }

        size_t beam_offset = (block_idx + 1) * max_beam_width;
        for (size_t i = 0; i < elem_count; i++) {
            // Remove backwards contribution from score
            (*prev_beam_front)[i].score -= float(block_back_scores[(*prev_beam_front)[i].state]);

            // Copy this new beam front into the beam persistent state
            beam_vector[beam_offset + i].state = (*prev_beam_front)[i].state;
            beam_vector[beam_offset + i].prev_element_index =
                    (*prev_beam_front)[i].prev_element_index;
            beam_vector[beam_offset + i].stay = (*prev_beam_front)[i].stay;
        }

        current_beam_width = elem_count;
    }

    // Extract final score
    const float final_score = (*prev_beam_front)[0].score;

    // Write out sequence bases and move table
    moves.resize(num_blocks);
    states.resize(num_blocks);

    // Note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = num_blocks; beam_idx != 0; beam_idx--) {
        size_t beam_addr = beam_idx * max_beam_width + element_index;
        states[beam_idx - 1] = int32_t(beam_vector[beam_addr].state);
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // Always step in the first event

    int hp_states[4] = {0, 0, 0,
                        0};  // What state index are the four homopolymers (A is always state 0)
    hp_states[3] = int(num_states) - 1;  // homopolymer T is always the last state. (11b per base)
    hp_states[1] = hp_states[3] / 3;     // calculate hp C from hp T (01b per base)
    hp_states[2] = hp_states[1] * 2;     // calculate hp G from hp C (10b per base)

    // Compute per-base qual data
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % num_bases;
        int base_to_emit = states[block_idx];

        // Compute a probability for this block, based on the path kmer. See the following explanation:
        // https://git.oxfordnanolabs.local/machine-learning/notebooks/-/blob/master/bonito-basecaller-qscores.ipynb
        const float* timestep_posts = posts + ((block_idx + 1) * num_states);

        // For states which are homopolymers, we don't want to count the states more than once
        bool is_hp = state == hp_states[0] || state == hp_states[1] || state == hp_states[2] ||
                     state == hp_states[3];
        float block_prob = float(timestep_posts[state]) * (is_hp ? -1.0f : 1.0f);

        // Add in left-shifted kmers
        int l_shift_idx = state / num_bases;
        int msb = int(num_states) / num_bases;
        for (int shift_base = 0; shift_base < num_bases; shift_base++) {
            block_prob += float(timestep_posts[l_shift_idx + msb * shift_base]);
        }

        // Add in the right-shifted kmers
        int r_shift_idx = (state * num_bases) % num_states;
        for (int shift_base = 0; shift_base < num_bases; shift_base++) {
            block_prob += float(timestep_posts[r_shift_idx + shift_base]);
        }
        block_prob = std::min(std::max(block_prob, 0.0f), 1.0f);  // clamp prob between 0 and 1
        block_prob = powf(block_prob, 0.4f);                      // Power fudge factor

        // Calculate a placeholder qscore for the "wrong" bases
        float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < num_bases; base++) {
            qual_data[block_idx * num_bases + base] =
                    (int(base) == base_to_emit ? block_prob : wrong_base_prob);
        }
    }

    return final_score;
}

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const torch::Tensor scores_t,
        const torch::Tensor back_guides_t,
        const torch::Tensor posts_t,
        size_t beam_width,
        float beam_cut,
        float fixed_stay_score,
        float q_shift,
        float q_scale,
        float temperature) {
    const int num_blocks = int(scores_t.size(0));
    const int num_states = get_num_states(scores_t.size(1));

    std::string sequence, qstring;
    std::vector<int32_t> states(num_blocks);
    std::vector<uint8_t> moves(num_blocks);
    std::vector<float> qual_data(num_blocks * num_bases);

    std::string type_str(scores_t.dtype().name());
    std::string post_type_str(posts_t.dtype().name());
    std::string guides_type_str(back_guides_t.dtype().name());

    if (post_type_str != type_str || guides_type_str != type_str) {
        throw std::runtime_error(
                "beam_search_decode: mismatched tensor types provided for posts, scores and "
                "guides");
    }

    // back guides and posts should be contiguous
    auto back_guides_contig = back_guides_t.expect_contiguous();
    auto posts_contig = posts_t.expect_contiguous();
    // scores_t may come from a tensor with chunks interleaved, but make sure the last dimension is contiguous
    auto scores_block_contig = (scores_t.stride(1) == 1) ? scores_t : scores_t.contiguous();
    size_t scores_block_stride = scores_block_contig.stride(0);
    if (type_str == "float") {
        const auto scores = scores_block_contig.data_ptr<float>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();

        beam_search<float>(scores, scores_block_stride, back_guides, posts, num_states, num_blocks,
                           beam_width, beam_cut, fixed_stay_score, states, moves, qual_data,
                           temperature);

    } else if (type_str == "signed char") {
        const auto scores = scores_block_contig.data_ptr<int8_t>();
        const auto back_guides = back_guides_contig->data_ptr<int8_t>();
        const auto fposts = ((posts_contig->to(torch::kFloat32) + 128.0f) / 255.0f);
        const auto posts = fposts.data_ptr<float>();

        beam_search<int8_t>(scores, scores_block_stride, back_guides, posts, num_states, num_blocks,
                            beam_width, beam_cut, fixed_stay_score, states, moves, qual_data,
                            temperature);

    } else {
        throw std::runtime_error(std::string("beam_search_decode: unsupported tensor type ") +
                                 type_str);
    }

    std::tie(sequence, qstring) = generate_sequence(moves, states, qual_data, q_shift, q_scale);

    return std::make_tuple(sequence, qstring, moves);
}
