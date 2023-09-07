#include "modbase/MotifMatcher.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "[modbase_motif_matcher]"

using std::make_tuple;

namespace {
const std::string SEQ = "AACCGGTTACGT";
}

TEST_CASE(TEST_GROUP ": test motifs", TEST_GROUP) {
    dorado::ModBaseModelConfig config;
    auto [motif, motif_offset, expected_results] =
            GENERATE(table<std::string, size_t, std::vector<size_t>>({
                    // clang-format off
                    make_tuple("CG", 0, std::vector<size_t>{3, 9}),    // C in CG
                    make_tuple("CG", 1, std::vector<size_t>{4, 10}),   // G in CG
                    make_tuple("C",  0, std::vector<size_t>{2, 3, 9}), // Any C
                    make_tuple("AA", 1, std::vector<size_t>{1}),       // A following A
                    make_tuple("TAC", 2, std::vector<size_t>{9}),      // C in TAC
                    make_tuple("X", 0, std::vector<size_t>{}),         // Invalid
                    // clang-format off
    }));

    CAPTURE(motif);
    CAPTURE(motif_offset);
    config.motif = motif;
    config.motif_offset = motif_offset;

    dorado::MotifMatcher matcher(config);
    auto hits = matcher.get_motif_hits(SEQ);
    CHECK(hits == expected_results);
}
