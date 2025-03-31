#include "splitter/myers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#define CUT_TAG "[myers]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(CUT_TAG " " name, CUT_TAG)

using dorado::splitter::EdistResult;
using dorado::splitter::myers_align;

DEFINE_TEST("Basic alignment, single hit") {
    const std::string_view query = "AAA";
    const std::string_view seq = "GGGCCCAAATTT";

    // The same location should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CATCH_CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    CATCH_REQUIRE(alignments.size() == 1);
    CATCH_CHECK(alignments[0].begin == 6);
    CATCH_CHECK(alignments[0].end == 9);
    CATCH_CHECK(alignments[0].edist == 0);
}

DEFINE_TEST("Basic alignment, multiple hits") {
    const std::string_view query = "CCC";
    const std::string_view seq = "GGGCCCAAATTTCCCGGG";

    // The same locations should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CATCH_CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    CATCH_REQUIRE(alignments.size() == 2);
    CATCH_CHECK(alignments[0].begin == 3);
    CATCH_CHECK(alignments[0].end == 6);
    CATCH_CHECK(alignments[0].edist == 0);
    CATCH_CHECK(alignments[1].begin == 12);
    CATCH_CHECK(alignments[1].end == 15);
    CATCH_CHECK(alignments[1].edist == 0);
}

DEFINE_TEST("Basic alignment, hit at end") {
    const std::string_view query = "TTT";
    const std::string_view seq = "GGGCCCAAATTT";

    // The same location should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CATCH_CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    CATCH_REQUIRE(alignments.size() == 1);
    CATCH_CHECK(alignments[0].begin == 9);
    CATCH_CHECK(alignments[0].end == 12);
    CATCH_CHECK(alignments[0].edist == 0);
}

DEFINE_TEST("Complex alignment, multiple hits") {
    const std::string_view query = "TACTTCGTTCAGTT";
    const std::string_view seq =
            "CTGTCGAGACCCTT"
            "TACTTCTTCTT"  // match 0
            "CACCAA"
            "TATTGTTATGTT"  // match 1
            "ATGTAGCC"
            "TGCTTCGTTCGGTT"  // match 2
            "ATGCGCCGCCAATATTAACCTCGGTAAAA"
            "TATCTTCGACCCAGTT"  // match 3
            "TTCGCGTCT";
    const auto max_edist = 4;

    const auto alignments = myers_align(query, seq, max_edist);
    CATCH_REQUIRE(alignments.size() == 4);
    CATCH_CHECK(alignments[0].begin == 14);
    CATCH_CHECK(alignments[0].end == 25);
    CATCH_CHECK(alignments[0].edist == 3);
    CATCH_CHECK(alignments[1].begin == 31);
    CATCH_CHECK(alignments[1].end == 43);
    CATCH_CHECK(alignments[1].edist == 4);
    CATCH_CHECK(alignments[2].begin == 51);
    CATCH_CHECK(alignments[2].end == 65);
    CATCH_CHECK(alignments[2].edist == 2);
    CATCH_CHECK(alignments[3].begin == 94);
    CATCH_CHECK(alignments[3].end == 110);
    CATCH_CHECK(alignments[3].edist == 4);
}

DEFINE_TEST("Complex alignment, doesn't crash when high edist near start") {
    const std::string_view query = "TACTTCGTTCAGTT";
    const std::string_view seq = "TTTTTTTTTTCTCCTGTTCTTGGTTCGGTTGT";
    const auto max_edist = 5;
    const auto alignments = myers_align(query, seq, max_edist);
    CATCH_CHECK(!alignments.empty());
}
