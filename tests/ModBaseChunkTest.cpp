#include "read_pipeline/ModBaseChunkCallerNode.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cstdint>
#include <tuple>
#include <utility>

#define TEST_GROUP "[modbase_chunk]"

using namespace dorado::modbase;
using Hits = std::vector<int64_t>;
using ChunkStart = std::pair<int64_t, int64_t>;
using ChunkStarts = std::vector<ChunkStart>;
using ScoreIdxs = std::vector<std::vector<int64_t>>;

namespace {

struct Params {
    std::string description;
    int64_t signal_len;
    int64_t chunk_size;
    int64_t context_samples_before;
    int64_t context_samples_after;
    bool end_align_last_chunk;
    int64_t scores_states;
    int64_t modbase_stride;
    Hits hits_to_sig;
    ChunkStarts expected_chunks;
    ScoreIdxs expected_scores;
};

void test_chunking(const Params & p) {
    // Get the chunk starts
    const ChunkStarts result_chunk_starts = dorado::ModBaseChunkCallerNode::get_chunk_starts(
            p.signal_len, p.hits_to_sig, p.chunk_size, p.context_samples_before,
            p.context_samples_after, p.end_align_last_chunk);

    CATCH_CHECK(p.expected_chunks == result_chunk_starts);

    // Resolve the modbase output indexes
    ScoreIdxs result_scores{p.expected_scores.size()};
    int chunk_idx = 0;
    for (const auto & [chunk_signal_start, hit_idx] : result_chunk_starts) {
        for (size_t hit = hit_idx; hit < p.hits_to_sig.size(); ++hit) {
            const auto hit_sig_abs = p.hits_to_sig.at(hit);
            const auto score_idx = dorado::ModBaseChunkCallerNode::resolve_score_index(
                    hit_sig_abs, chunk_signal_start, p.scores_states, p.chunk_size,
                    p.context_samples_before, p.context_samples_after, p.modbase_stride);

            // To save repeatedly multiplying the test params by the scores_states down-scale here
            const auto downscaled = score_idx > 0 ? score_idx / p.scores_states : score_idx;
            result_scores.at(chunk_idx).emplace_back(downscaled);
        }
        ++chunk_idx;
    }

    CATCH_CHECK(p.expected_scores.size() == result_scores.size());
    for (size_t chunk_i = 0; chunk_i < result_scores.size(); ++chunk_i) {
        CATCH_CAPTURE(chunk_i);
        CATCH_CHECK(p.expected_scores.at(chunk_i) == result_scores.at(chunk_i));
    }
}

CATCH_TEST_CASE(TEST_GROUP ": chunking not stride-aligned", TEST_GROUP) {
    // Setting modbase stride to 1 allows us to use "easy" un-normalised values which would otherwise
    // be invalid as they're not stride aligned.
    constexpr int64_t no_stride = 1;
    constexpr bool no_align = false;
    // clang-format off
    auto [p] = GENERATE_COPY(table<Params>({
            Params{"Single chunk - capture whole read", 
                20, 100, 0, 0, no_align, 3, no_stride, {0, 10, 19},
                {{0, 0}},
                {{0, 10, 19}}
            },
            Params{"Single chunk - 0 context samples landing on last sample", 
                20, 20, 0, 0, no_align, 2, no_stride, {0, 10, 19},
                {{0, 0}},
                {{0, 10, 19}}
            },
            Params{"Single chunk - non zero start - zero context", 
                20, 20, 0, 0, no_align, 3, no_stride, {10, 19},
                {{10, 0}},
                {{0, 9}}
            },
            Params{"Single chunk - short signal length", 
                20, 20, 0, 0, no_align, 2, no_stride, {0, 10, 50},
                {{0, 0}},
                {{0, 10, -2}}
            },
            Params{"5 lead-in context samples 1 - 2 chunks", 
                20, 20, 5, 5, no_align, 3, no_stride, {0, 10, 19},
                {{0, 0}, {14, 2}},
                {{0, 10, -2}, {5}}
            },
            Params{"5 lead-in context samples 2 - 3 chunks",
                40, 20, 5, 5, no_align, 3, no_stride, {0, 10, 20, 30, 39},
                {{0, 0}, {15, 2}, {34, 4}},
                {{0, 10, -2, -2, -2}, {5, 15, -2}, {5}}
            },
            Params{"5 lead-in context samples 3 - non zero start",
                40, 20, 5, 5, no_align, 3, no_stride, {9, 10, 20, 30},
                {{4, 0}, {15, 2}},
                {{5, 6, -2, -2}, {5, 15}}
            },
            Params{"5 lead-in context samples 4 - non zero start - last hit not caught",
                40, 20, 5, 5, no_align, 3, no_stride, {9, 10, 20, 31},
                {{4, 0}, {15, 2}, {26, 3}},
                {{5, 6, -2, -2}, {5, -2}, {5}}
            },
            Params{"10 lead-in context samples 1", 
                20, 20, 10, 10, no_align, 3, no_stride, {0, 10, 19},
                {{0, 0}, {9, 2}},
                {{0, 10, -2}, {10}}
            },
            Params{"2 sparse chunks", 
                200, 20, 10, 10,  no_align, 3, no_stride, {50, 150},
                {{40, 0}, {140, 1}},
                {{10, -2}, {10}}
            },
            Params{"Large chunk asymmetric context", 
                800, 100, 20, 10,  no_align, 3, no_stride, {88, 92, 512, 555, 671, 700, 789},
                {{68, 0}, {492, 2}, {651, 4}, {769, 6}},
                {{20, 24, -2, -2, -2, -2, -2}, {20, 63, -2, -2, -2}, {20, 49, -2}, {20}}
            },
    }));
    // clang-format on
    CATCH_SECTION(p.description) { test_chunking(p); }
}

CATCH_TEST_CASE(TEST_GROUP ": chunking not stride-aligned end-aligned", TEST_GROUP) {
    // Setting modbase stride to 1 allows us to use "easy" un-normalised values which would otherwise
    // be invalid as they're not stride aligned.
    constexpr int64_t no_stride = 1;
    constexpr bool align = true;
    // clang-format off
    auto [p] = GENERATE_COPY(table<Params>({
            Params{"Single chunk - capture whole read", 
                20, 100, 0, 0, align, 3, no_stride, {0, 10, 19},
                {{0, 0}},
                {{0, 10, 19}}
            },
            Params{"Single chunk - 0 context samples landing on last sample", 
                20, 20, 0, 0, align, 2, no_stride, {0, 10, 19},
                {{0, 0}},
                {{0, 10, 19}}
            },
            Params{"Single chunk - non zero start - zero context", 
                20, 20, 0, 0, align, 3, no_stride, {10, 19},
                {{10, 0}},
                {{0, 9}}
            },
            Params{"Single chunk - short signal length", 
                20, 20, 0, 0, align, 2, no_stride, {0, 10, 50},
                {{0, 0}},
                {{0, 10, -2}}
            },
            Params{"5 lead-in context samples 1 - 2 chunks", 
                20, 20, 5, 5, align, 3, no_stride, {0, 10, 19},
                {{0, 0}, {4, 2}},
                {{0, 10, -2}, {15}}
            },
            Params{"5 lead-in context samples 2 - 3 chunks",
                40, 20, 5, 5, align, 3, no_stride, {0, 10, 20, 30, 39},
                {{0, 0}, {15, 2}, {24, 4}},
                {{0, 10, -2, -2, -2}, {5, 15, -2}, {15}}
            },
            Params{"5 lead-in context samples 3 - non zero start",
                40, 20, 5, 5, align, 3, no_stride, {9, 10, 20, 30},
                {{4, 0}, {15, 2}},
                {{5, 6, -2, -2}, {5, 15}}
            },
            Params{"5 lead-in context samples 4 - non zero start - last hit not caught",
                40, 20, 5, 5, align, 3, no_stride, {9, 10, 20, 31},
                {{4, 0}, {15, 2}, {16, 3}},
                {{5, 6, -2, -2}, {5, -2}, {15}}
            },
            Params{"10 lead-in context samples 1", 
                20, 20, 10, 10, align, 3, no_stride, {0, 10, 19},
                {{0, 0}, {9, 2}},
                {{0, 10, -2}, {10}}
            },
            Params{"2 sparse chunks", 
                200, 20, 10, 10,  align, 3, no_stride, {50, 150},
                {{40, 0}, {140, 1}},
                {{10, -2}, {10}}
            },
            Params{"Large chunk asymmetric context", 
                800, 100, 20, 10,  align, 3, no_stride, {88, 92, 512, 555, 671, 700, 789},
                {{68, 0}, {492, 2}, {651, 4}, {699, 6}},
                {{20, 24, -2, -2, -2, -2, -2}, {20, 63, -2, -2, -2}, {20, 49, -2}, {90}}
            },
    }));
    // clang-format on
    CATCH_SECTION(p.description) { test_chunking(p); }
}

CATCH_TEST_CASE(TEST_GROUP ": chunking stride-aligned", TEST_GROUP) {
    // clang-format off
    // Use the modbase stride correctly
    constexpr int64_t stride = 3;
    constexpr bool no_align = false;

    auto [p] = GENERATE_COPY(table<Params>({
            Params{"Single chunk - capture whole read", 
                20, 100, 0, 0, no_align, 2, stride, {0, 9, 18},
                {{0, 0}},
                {{0, 3, 6}}
            },
            Params{"2 sparse chunks", 
                200, 24, 12, 12,  no_align, 3, stride, {60, 180},
                {{48, 0}, {168, 1}},
                {{4, -2}, {4}}
            },
    }));
    // clang-format on
    CATCH_SECTION(p.description) { test_chunking(p); }
}

CATCH_TEST_CASE(TEST_GROUP ": chunking context-centered", TEST_GROUP) {
    // clang-format off
    constexpr int64_t stride = 3;
    constexpr bool no_align = false;

    auto [p] = GENERATE_COPY(table<Params>({
            // Chunk size == context size so all hits are context-centered chunks
            Params{"5mCG context size", 
                3000, 204, 102, 102, no_align, 3, stride, {1800, 1820, 1902},
                {{1698, 0}, {1718, 1}, {1800, 2}},
                {{34, -2, -2}, {34, -2}, {34} }
            },
            Params{"6mA context size", 
                3000, 300, 150, 150, no_align, 3, stride, {1800, 1820, 1902},
                {{1650, 0}, {1670, 1}, {1752, 2}},
                {{50, -2, -2}, {50, -2}, {50} }
            },
    }));
    // clang-format on
    CATCH_SECTION(p.description) { test_chunking(p); }
}

CATCH_TEST_CASE(TEST_GROUP ": resolve_score_index", TEST_GROUP) {
    auto [expected, hit_sig_abs, chunk_signal_start, scores_states, chunk_size,
          context_samples_before, context_samples_after, modbase_stride] =
            // clang-format off
            GENERATE(table<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>({
                    // Happy cases
                    std::make_tuple(0, 0, 0, 1, 1, 0, 1, 1),
                    std::make_tuple(2, 2, 0, 1, 3, 0, 1, 1),
                    std::make_tuple(12, 12, 0, 1, 30, 0, 1, 1),
                    std::make_tuple(23000, 33000, 10000, 1, 30000, 0, 1, 1),
                    // Happy strided
                    std::make_tuple(4, 12, 0, 1, 30, 0, 1, 3),
                    std::make_tuple(9, 27, 0, 1, 30, 0, 1, 3),
                    std::make_tuple(1, 6, 0, 1, 30, 0, 1, 6),
                    // Multiple scores states
                    std::make_tuple(4, 2, 0, 2, 3, 0, 1, 1),
                    std::make_tuple(6, 2, 0, 3, 3, 0, 1, 1),
                    std::make_tuple(24, 12, 0, 2, 30, 0, 1, 1),
                    std::make_tuple(18, 27, 0, 2, 30, 0, 1, 3),
                    // Out of bounds
                    std::make_tuple(-2, 2, 1, 1, 1, 0, 1, 1),
                    std::make_tuple(-2, 30, 0, 1, 30, 0, 1, 3),
                    // Handling hits in the context samples after
                    std::make_tuple(-2, 4, 0, 1, 5, 0, 3, 1),    
                    std::make_tuple(-2, 90, 0, 1, 100, 0, 20, 1),                    
                    // Handling hits within context samples before -
                    std::make_tuple(-1, 15, 12, 1, 30, 11, 1, 1),
                    std::make_tuple(-1, 100, 90, 1, 30, 20, 1, 3),
            }));
    // clang-format on
    const int64_t result = dorado::ModBaseChunkCallerNode::resolve_score_index(
            hit_sig_abs, chunk_signal_start, scores_states, chunk_size, context_samples_before,
            context_samples_after, modbase_stride);

    CATCH_CAPTURE(result, expected, hit_sig_abs, chunk_signal_start, scores_states, chunk_size,
                  context_samples_before, context_samples_after, modbase_stride);
    CATCH_CHECK(expected == result);
}

CATCH_TEST_CASE(TEST_GROUP ": resolve_score_index exceptions", TEST_GROUP) {
    const std::string hit_before_start = "Modbase hit before chunk start.";
    const std::string not_stride_aligned = "Modbase score did not align to canonical base.";

    auto [hit_sig_abs, chunk_signal_start, scores_states, chunk_size, context_samples_before,
          context_samples_after, modbase_stride, expected_msg] =
            // clang-format off
            GENERATE_COPY(table<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, std::string>({
                std::make_tuple(-1, 0, 1, 1, 0, 1, 1, hit_before_start),
                std::make_tuple(0, 1, 1, 1, 0, 1, 1, hit_before_start),
                std::make_tuple(999, 1000, 1, 1, 0, 1, 1, hit_before_start),
                std::make_tuple(2, 1, 1, 10, 0, 1, 3, not_stride_aligned),
                std::make_tuple(3, 1, 1, 10, 0, 1, 3, not_stride_aligned),
            }));
    
    CATCH_CAPTURE(hit_sig_abs, chunk_signal_start, scores_states, chunk_size,
            context_samples_before, context_samples_after, modbase_stride);
    
    CATCH_CHECK_THROWS_WITH(dorado::ModBaseChunkCallerNode::resolve_score_index(
            hit_sig_abs, chunk_signal_start, scores_states, chunk_size, context_samples_before,
            context_samples_after, modbase_stride), expected_msg);

}

}  // namespace