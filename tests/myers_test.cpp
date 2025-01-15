#include "splitter/myers.h"

#include <catch2/catch_all.hpp>

#define CUT_TAG "[myers]"
#define DEFINE_TEST(name) TEST_CASE(CUT_TAG " " name, CUT_TAG)

using dorado::splitter::EdistResult;
using dorado::splitter::myers_align;

DEFINE_TEST("Basic alignment, single hit") {
    const std::string_view query = "AAA";
    const std::string_view seq = "GGGCCCAAATTT";

    // The same location should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    REQUIRE(alignments.size() == 1);
    CHECK(alignments[0].begin == 6);
    CHECK(alignments[0].end == 9);
    CHECK(alignments[0].edist == 0);
}

DEFINE_TEST("Basic alignment, multiple hits") {
    const std::string_view query = "CCC";
    const std::string_view seq = "GGGCCCAAATTTCCCGGG";

    // The same locations should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    REQUIRE(alignments.size() == 2);
    CHECK(alignments[0].begin == 3);
    CHECK(alignments[0].end == 6);
    CHECK(alignments[0].edist == 0);
    CHECK(alignments[1].begin == 12);
    CHECK(alignments[1].end == 15);
    CHECK(alignments[1].edist == 0);
}

DEFINE_TEST("Basic alignment, hit at end") {
    const std::string_view query = "TTT";
    const std::string_view seq = "GGGCCCAAATTT";

    // The same location should be hit for all edists.
    const auto max_edist = GENERATE(0, 1, 2);
    CAPTURE(max_edist);

    const auto alignments = myers_align(query, seq, max_edist);
    REQUIRE(alignments.size() == 1);
    CHECK(alignments[0].begin == 9);
    CHECK(alignments[0].end == 12);
    CHECK(alignments[0].edist == 0);
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
    REQUIRE(alignments.size() == 4);
    CHECK(alignments[0].begin == 14);
    CHECK(alignments[0].end == 25);
    CHECK(alignments[0].edist == 3);
    CHECK(alignments[1].begin == 31);
    CHECK(alignments[1].end == 43);
    CHECK(alignments[1].edist == 4);
    CHECK(alignments[2].begin == 51);
    CHECK(alignments[2].end == 65);
    CHECK(alignments[2].edist == 2);
    CHECK(alignments[3].begin == 94);
    CHECK(alignments[3].end == 110);
    CHECK(alignments[3].edist == 4);
}

DEFINE_TEST("Complex alignment, doesn't crash when high edist near start") {
    const std::string_view query = "TACTTCGTTCAGTT";
    const std::string_view seq = "TTTTTTTTTTCTCCTGTTCTTGGTTCGGTTGT";
    const auto max_edist = 5;
    const auto alignments = myers_align(query, seq, max_edist);
    CHECK(!alignments.empty());
}
