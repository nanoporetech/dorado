#include "beam_search.h"

#include "utils/simd.h"

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <math.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

namespace {

// 16 bit state supports 7-mers with 4 bases.
typedef uint16_t state_t;

constexpr int NUM_BASE_BITS = 2;
constexpr int NUM_BASES = 1 << NUM_BASE_BITS;

// This is the data we need to retain for the whole beam
struct BeamElement {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

// This is the data we need to retain for only the previous timestep (block) in the beam
// (and what we construct for the new timestep)
struct BeamFrontElement {
    uint32_t hash;
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

float log_sum_exp(float x, float y) {
    float abs_diff = std::abs(x - y);
    return std::max(x, y) + ((abs_diff < 17.0f) ? (std::log1p(std::exp(-abs_diff))) : 0.0f);
}

int get_num_states(size_t num_trans_states) {
    if (num_trans_states % NUM_BASES != 0) {
        throw std::runtime_error("Unexpected number of transition states in beam search decode.");
    }
    return int(num_trans_states / NUM_BASES);
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
    std::array<char, 4> alphabet = {'A', 'C', 'G', 'T'};
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
        baseProbs[i] = -10.0f * std::log10(baseProbs[i]);
        float qscore = baseProbs[i] * scale + shift;
        qscore = std::clamp(qscore, 1.0f, 50.0f);
        qstring[i] = static_cast<char>(33.5f + qscore);
    }

    return make_tuple(std::move(sequence), std::move(qstring));
}

// Incorporates NUM_NEW_BITS into a Castagnoli CRC32, aka CRC32C
// (not the same polynomial as CRC32 as used in zip/ethernet).
template <int NUM_NEW_BITS>
uint32_t crc32c(uint32_t crc, uint32_t new_bits) {
    // Note that this is the reversed polynomial.
    constexpr uint32_t POLYNOMIAL = 0x82f63b78u;
    for (int i = 0; i < NUM_NEW_BITS; ++i) {
        auto b = (new_bits ^ crc) & 1;
        crc >>= 1;
        if (b) {
            crc ^= POLYNOMIAL;
        }
        new_bits >>= 1;
    }
    return crc;
}

}  // anonymous namespace

namespace dorado::basecall::decode {

template <typename T, typename U>
float beam_search(const T* const scores,
                  size_t scores_block_stride,
                  const float* const back_guide,
                  const U* const posts,
                  int num_state_bits,
                  size_t num_blocks,
                  size_t max_beam_width,
                  float beam_cut,
                  float fixed_stay_score,
                  std::vector<int32_t>& states,
                  std::vector<uint8_t>& moves,
                  std::vector<float>& qual_data,
                  float score_scale,
                  float posts_scale) {
    const size_t num_states = 1ull << num_state_bits;
    const auto states_mask = static_cast<state_t>(num_states - 1);

    if (max_beam_width > 256) {
        throw std::range_error("Beamsearch max_beam_width cannot be greater than 256.");
    }

    // Some values we need
    constexpr uint32_t CRC_SEED = 0x12345678u;
    const float log_beam_cut =
            (beam_cut > 0.0f) ? logf(beam_cut) : std::numeric_limits<float>::max();

    // Create the beam.  We need to keep beam_width elements for each block, plus the initial state
    std::vector<BeamElement> beam_vector(max_beam_width * (num_blocks + 1));

    // Create the previous and current beam fronts
    // Each existing element can be extended by one of NUM_BASES, or be a stay.
    size_t max_beam_candidates = (NUM_BASES + 1) * max_beam_width;

    std::vector<BeamFrontElement> current_beam_front(max_beam_candidates);
    std::vector<BeamFrontElement> prev_beam_front(max_beam_candidates);

    std::vector<float> current_scores(max_beam_candidates);
    std::vector<float> prev_scores(max_beam_candidates);

    // Find the score an initial element needs in order to make it into the beam
    T beam_init_threshold = std::numeric_limits<T>::lowest();
    if (max_beam_width < num_states) {
        // Copy the first set of back guides and sort to extract max_beam_width highest elements
        std::vector<T> sorted_back_guides(num_states);
        std::memcpy(sorted_back_guides.data(), back_guide, num_states * sizeof(T));

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
            prev_beam_front[beam_element] = {crc32c<32>(CRC_SEED, uint32_t(state)),
                                             static_cast<state_t>(state), 0, false};
            prev_scores[beam_element] = 0.0f;
            ++beam_element;
        }
    }

    // Copy this initial beam front into the beam persistent state
    size_t current_beam_width = std::min(max_beam_width, num_states);
    for (size_t element_idx = 0; element_idx < current_beam_width; ++element_idx) {
        beam_vector[element_idx].state = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index =
                (prev_beam_front)[element_idx].prev_element_index;
        beam_vector[element_idx].stay = prev_beam_front[element_idx].stay;
    }

    // Iterate through blocks, extending beam
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const T* const block_scores = scores + (block_idx * scores_block_stride);
        // Retrieves the given score as a float, multiplied by score_scale.
        const auto fetch_block_score = [block_scores, score_scale](size_t idx) {
            return static_cast<float>(block_scores[idx]) * score_scale;
        };
        const float* const block_back_scores = back_guide + ((block_idx + 1) << num_state_bits);

        /*  kmer transitions order:
         *  N^K , N array
         *  Elements stored as resulting kmer and modifying action (stays have a fixed score and are not computed).
         *  Kmer index is lexicographic with most recent base in the fastest index
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

        float max_score = std::numeric_limits<float>::lowest();

        // Essentially a k=1 Bloom filter, indicating the presence of steps with particular
        // sequence hashes.  Avoids comparing stay hashes against all possible progenitor
        // states where none of them has the requisite sequence hash.
        const uint32_t HASH_PRESENT_BITS = 4096;
        const uint32_t HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1;
        std::bitset<HASH_PRESENT_BITS> step_hash_present;  // Default constructor zeros content.

        // Generate list of candidate elements for this timestep (block).
        // As we do so, update the maximum score.
        size_t new_elem_count = 0;
        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const auto& previous_element = prev_beam_front[prev_elem_idx];

            // Expand all the possible steps
            for (int new_base = 0; new_base < NUM_BASES; new_base++) {
                state_t new_state =
                        (state_t((previous_element.state << NUM_BASE_BITS) & states_mask) |
                         state_t(new_base));
                const auto move_idx = static_cast<state_t>(
                        (new_state << NUM_BASE_BITS) +
                        (((previous_element.state << NUM_BASE_BITS) >> num_state_bits)));
                float new_score = prev_scores[prev_elem_idx] + fetch_block_score(move_idx) +
                                  static_cast<float>(block_back_scores[new_state]);
                uint32_t new_hash = crc32c<NUM_BASE_BITS>(previous_element.hash, new_base);

                step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

                // Add new element to the candidate list
                current_beam_front[new_elem_count] = {new_hash, new_state, (uint8_t)prev_elem_idx,
                                                      false};
                current_scores[new_elem_count] = new_score;
                max_score = std::max(max_score, new_score);
                ++new_elem_count;
            }
        }

        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const auto& previous_element = prev_beam_front[prev_elem_idx];
            // Add the possible stay.
            const float stay_score = prev_scores[prev_elem_idx] + fixed_stay_score +
                                     static_cast<float>(block_back_scores[previous_element.state]);
            current_beam_front[new_elem_count] = {previous_element.hash, previous_element.state,
                                                  (uint8_t)prev_elem_idx, true};
            current_scores[new_elem_count] = stay_score;
            max_score = std::max(max_score, stay_score);

            // Determine whether the path including this stay duplicates another sequence ending in
            // a step.
            if (step_hash_present[previous_element.hash & HASH_PRESENT_MASK]) {
                size_t stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;
                // latest base is in smallest bits
                int stay_latest_base = int(previous_element.state & 3);

                // Go through all the possible step extensions that match this destination base with the stay and compare
                // their hashes, merging if we find any.
                for (size_t prev_elem_comp_idx = 0; prev_elem_comp_idx < current_beam_width;
                     prev_elem_comp_idx++) {
                    size_t step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base;
                    if (current_beam_front[stay_elem_idx].hash ==
                        current_beam_front[step_elem_idx].hash) {
                        if (current_scores[stay_elem_idx] > current_scores[step_elem_idx]) {
                            // Fold the step into the stay
                            const float folded_score = log_sum_exp(current_scores[stay_elem_idx],
                                                                   current_scores[step_elem_idx]);
                            current_scores[stay_elem_idx] = folded_score;
                            max_score = std::max(max_score, folded_score);
                            // The step element will end up last, sorted by score
                            current_scores[step_elem_idx] = std::numeric_limits<float>::lowest();
                        } else {
                            // Fold the stay into the step
                            const float folded_score = log_sum_exp(current_scores[stay_elem_idx],
                                                                   current_scores[step_elem_idx]);
                            current_scores[step_elem_idx] = folded_score;
                            max_score = std::max(max_score, folded_score);
                            // The stay element will end up last, sorted by score
                            current_scores[stay_elem_idx] = std::numeric_limits<float>::lowest();
                        }
                    }
                }
            }

            ++new_elem_count;
        }

        // Starting point for finding the cutoff score is the beam cut score
        float beam_cutoff_score = max_score - log_beam_cut;

        auto get_elem_count = [new_elem_count, &beam_cutoff_score, &current_scores]() {
            // Count the elements which meet the beam cutoff.
            size_t elem_count = 0;
            const float* score_ptr = current_scores.data();
#if !ENABLE_NEON_IMPL
            for (int i = int(new_elem_count); i; --i) {
                if (*score_ptr >= beam_cutoff_score) {
                    ++elem_count;
                }
                ++score_ptr;
            }
#else
            uint32x4_t counts_x4_a = vdupq_n_u32(0u);
            uint32x4_t counts_x4_b = vdupq_n_u32(0u);
            const float32x4_t cutoff_x4 = vdupq_n_f32(beam_cutoff_score);

            // 8 fold unrolled version has the small upside that both loads
            // can be done with a single ldp instruction.
            const int kUnroll = 8;
            for (int i = int(new_elem_count) / kUnroll; i; --i) {
                // True comparison sets lane bits to 0xffffffff, or -1 in two's complement,
                // which we subtract to increment our counts.
                float32x4_t scores_x4_a = vld1q_f32(score_ptr);
                uint32x4_t comparisons_x4_a = vcgeq_f32(scores_x4_a, cutoff_x4);
                counts_x4_a = vsubq_u32(counts_x4_a, comparisons_x4_a);

                float32x4_t scores_x4_b = vld1q_f32(score_ptr + 4);
                uint32x4_t comparisons_x4_b = vcgeq_f32(scores_x4_b, cutoff_x4);
                counts_x4_b = vsubq_u32(counts_x4_b, comparisons_x4_b);

                score_ptr += 8;
            }
            // Add together the result of 2 horizontal adds.
            elem_count = vaddvq_u32(counts_x4_a) + vaddvq_u32(counts_x4_b);
            for (int i = new_elem_count % kUnroll; i; --i) {
                if (*score_ptr >= beam_cutoff_score) {
                    ++elem_count;
                }
                ++score_ptr;
            }
#endif
            return elem_count;
        };

        // Count the elements which meet the min score
        size_t elem_count = get_elem_count();

        if (elem_count > max_beam_width) {
            // Need to find a score which doesn't return too many scores, but doesn't reduce beam width too much
            size_t min_beam_width =
                    (max_beam_width * 8) / 10;  // 80% of beam width is the minimum we accept.
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            constexpr int MAX_GUESSES = 10;
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
                elem_count = get_elem_count();
                ++num_guesses;
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
                elem_count = get_elem_count();
            }

            // Clamp the element count to the max beam width in case of failure 2 from above.
            elem_count = std::min(elem_count, max_beam_width);
        }

        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < new_elem_count; ++read_idx) {
            if (current_scores[read_idx] >= beam_cutoff_score) {
                if (write_idx < max_beam_width) {
                    prev_beam_front[write_idx] = current_beam_front[read_idx];
                    prev_scores[write_idx] = current_scores[read_idx];
                    ++write_idx;
                } else {
                    break;
                }
            }
        }

        // At the last timestep, we need to ensure the best path corresponds to element 0.
        // The other elements don't matter.
        if (block_idx == num_blocks - 1) {
            float best_score = std::numeric_limits<float>::lowest();
            size_t best_score_index = 0;
            for (size_t i = 0; i < elem_count; i++) {
                if (prev_scores[i] > best_score) {
                    best_score = prev_scores[i];
                    best_score_index = i;
                }
            }
            std::swap(prev_beam_front[0], prev_beam_front[best_score_index]);
            std::swap(prev_scores[0], prev_scores[best_score_index]);
        }

        size_t beam_offset = (block_idx + 1) * max_beam_width;
        for (size_t i = 0; i < elem_count; ++i) {
            // Remove backwards contribution from score
            prev_scores[i] -= float(block_back_scores[prev_beam_front[i].state]);

            // Copy this new beam front into the beam persistent state
            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }

        current_beam_width = elem_count;
    }

    // Extract final score
    const float final_score = prev_scores[0];

    // Write out sequence bases and move table
    moves.resize(num_blocks);
    states.resize(num_blocks);

    // Note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = num_blocks; beam_idx != 0; --beam_idx) {
        size_t beam_addr = beam_idx * max_beam_width + element_index;
        states[beam_idx - 1] = int32_t(beam_vector[beam_addr].state);
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // Always step in the first event

    int shifted_states[2 * NUM_BASES];

    // Compute per-base qual data
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % NUM_BASES;
        int base_to_emit = states[block_idx];

        // Compute a probability for this block, based on the path kmer. See the following explanation:
        // https://git.oxfordnanolabs.local/machine-learning/notebooks/-/blob/master/bonito-basecaller-qscores.ipynb
        const U* const timestep_posts = posts + ((block_idx + 1) << num_state_bits);
        const auto fetch_post = [timestep_posts, posts_scale](size_t idx) {
            return static_cast<float>(timestep_posts[idx]) * posts_scale;
        };

        float block_prob = fetch_post(state);

        // Get indices of left- and right-shifted kmers
        int l_shift_idx = state >> NUM_BASE_BITS;
        int r_shift_idx = (state << NUM_BASE_BITS) % num_states;
        int msb = int(num_states) >> NUM_BASE_BITS;
        int l_shift_state, r_shift_state;
        for (int shift_base = 0; shift_base < NUM_BASES; ++shift_base) {
            l_shift_state = l_shift_idx + msb * shift_base;
            shifted_states[2 * shift_base] = l_shift_state;

            r_shift_state = r_shift_idx + shift_base;
            shifted_states[2 * shift_base + 1] = r_shift_state;
        }

        // Add probabilities for unique states
        int candidate_state;
        for (size_t state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
            candidate_state = shifted_states[state_idx];
            // don't double-count this shifted state if it matches the current state
            bool count_state = (candidate_state != state);
            // or any other shifted state that we've seen so far
            if (count_state) {
                for (size_t inner_state = 0; inner_state < state_idx; ++inner_state) {
                    if (shifted_states[inner_state] == candidate_state) {
                        count_state = false;
                        break;
                    }
                }
            }
            if (count_state) {
                block_prob += fetch_post(candidate_state);
            }
        }

        block_prob = std::clamp(block_prob, 0.0f, 1.0f);
        block_prob = std::pow(block_prob, 0.4f);  // Power fudge factor

        // Calculate a placeholder qscore for the "wrong" bases
        const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] =
                    (int(base) == base_to_emit ? block_prob : wrong_base_prob);
        }
    }

    return final_score;
}

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const at::Tensor& scores_t,
        const at::Tensor& back_guides_t,
        const at::Tensor& posts_t,
        size_t max_beam_width,
        float beam_cut,
        float fixed_stay_score,
        float q_shift,
        float q_scale,
        float byte_score_scale) {
    const int num_blocks = int(scores_t.size(0));
    const int num_states = get_num_states(scores_t.size(1));
    const int num_state_bits = static_cast<int>(std::log2(num_states));
    if (1 << num_state_bits != num_states) {
        throw std::runtime_error("num_states must be an integral power of 2");
    }

    // Back guides must be floats regardless of scores type.
    if (back_guides_t.dtype() != at::ScalarType::Float) {
        throw std::runtime_error("beam_search_decode: back guides type must be float");
    }

    // back guides and posts should be contiguous
    auto back_guides_contig = back_guides_t.expect_contiguous();
    auto posts_contig = posts_t.expect_contiguous();
    // scores_t may come from a tensor with chunks interleaved, but make sure the last dimension is contiguous
    auto scores_block_contig = (scores_t.stride(1) == 1) ? scores_t : scores_t.contiguous();

    std::vector<int32_t> states(num_blocks);
    std::vector<uint8_t> moves(num_blocks);
    std::vector<float> qual_data(num_blocks * NUM_BASES);

    const size_t scores_block_stride = scores_block_contig.stride(0);
    if (scores_t.dtype() == at::ScalarType::Float) {
        // If the scores are floats, so must the other tensors.
        if (posts_t.dtype() != at::ScalarType::Float) {
            throw std::runtime_error(
                    "beam_search_decode: only float posts are supported for float scores");
        }

        const auto scores = scores_block_contig.data_ptr<float>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();

        beam_search<float, float>(scores, scores_block_stride, back_guides, posts, num_state_bits,
                                  num_blocks, max_beam_width, beam_cut, fixed_stay_score, states,
                                  moves, qual_data, 1.0f, 1.0f);
    } else if (scores_t.dtype() == at::kChar) {
        // If the scores are 8 bit, the posterior probabilities must be 16 bit (Apple path).
        if (posts_t.dtype() != at::ScalarType::Short) {
            throw std::runtime_error(
                    "beam_search_decode: only int16 posts are supported for int8 scores");
        }

        const auto scores = scores_block_contig.data_ptr<int8_t>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<int16_t>();
        const float posts_scale = static_cast<float>(1.0 / 32767.0);
        beam_search<int8_t, int16_t>(scores, scores_block_stride, back_guides, posts,
                                     num_state_bits, num_blocks, max_beam_width, beam_cut,
                                     fixed_stay_score, states, moves, qual_data, byte_score_scale,
                                     posts_scale);

    } else if (scores_t.dtype() == at::kHalf) {
        if (posts_t.dtype() != at::ScalarType::Float) {
            throw std::runtime_error(
                    "beam_search_decode: only float32 posts are supported for int16 scores");
        }

        const auto scores = scores_block_contig.data_ptr<c10::Half>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();
        beam_search<c10::Half, float>(scores, scores_block_stride, back_guides, posts,
                                      num_state_bits, num_blocks, max_beam_width, beam_cut,
                                      fixed_stay_score, states, moves, qual_data, 1.0f, 1.0f);

    } else {
        throw std::runtime_error(std::string("beam_search_decode: unsupported tensor type ") +
                                 std::string(scores_t.dtype().name()));
    }

    auto [sequence, qstring] = generate_sequence(moves, states, qual_data, q_shift, q_scale);

    return {std::move(sequence), std::move(qstring), std::move(moves)};
}

}  // namespace dorado::basecall::decode
