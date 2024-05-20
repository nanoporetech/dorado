#include "alignment/sam_utils.h"

#include <catch2/catch.hpp>

#include <algorithm>
#include <numeric>
#include <sstream>

#define CUT_TAG "[sam_utils]"

using namespace dorado;
using namespace dorado::alignment;

TEST_CASE(CUT_TAG " test parse cigar", CUT_TAG) {
    std::string cigar = "13I15D17M13D1I6M48I";
    AlignmentResult alres;
    parse_cigar(cigar, alres);
    CHECK(13 + 1 + 48 == alres.num_insertions);
    CHECK(15 + 13 == alres.num_deletions);
    CHECK(17 + 6 == alres.num_aligned);
}

TEST_CASE(CUT_TAG " test parse", CUT_TAG) {
    std::string fasta =
            "@SQ\tSN:ref1\tLN:100\n"
            "@SQ\tSN:ref2\tLN:500\n"
            "read_1\t0\tref1\t50\t12\t12H20M2I10M1D5H\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTACGT\t*"
            "\tNM:i:5\tAS:i:100";

    auto results = parse_sam_lines(fasta, "", "");
    REQUIRE(results.size() == 1);

    AlignmentResult alres = results[0];

    CHECK(std::string("ref1") == alres.genome);
    CHECK(50 == alres.genome_start);
    CHECK(81 == alres.genome_end);
    CHECK(32 == alres.num_events);
    CHECK(12 == alres.strand_start);
    CHECK(44 == alres.strand_end);
    CHECK(2 == alres.num_insertions);
    CHECK(1 == alres.num_deletions);
    CHECK(30 == alres.num_aligned);
    CHECK(28 == alres.num_correct);
    CHECK_THAT(alres.identity, Catch::Matchers::WithinAbs(28.0f / 30, 0.00001f));
    CHECK_THAT(alres.accuracy, Catch::Matchers::WithinAbs(28.0f / (30 + 1 + 2), 0.00001f));
    CHECK(100 == alres.strand_score);
}

TEST_CASE(CUT_TAG " test parse coverage", CUT_TAG) {
    std::string fasta = GENERATE(
            // Test we get a too low Coverage return value if sequenceLength is the limiting factor
            "@SQ\tSN:ref1\tLN:1000\n"
            "@SQ\tSN:ref2\tLN:500\n"
            "read_1\t0\tref1\t50\t12\t12H20M2I10M1D5H\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTACGT\t*"
            "\tNM:i:5\tAS:i:100",

            // Test we get a too low Coverage return value if referenceLength is the limiting factor
            "@SQ\tSN:ref1\tLN:50\n"
            "@SQ\tSN:ref2\tLN:500\n"
            "read_1\t0\tref1\t50\t12\t12H20M2I10M1D5H\t*\t0\t0\tACGTACGTACGTACGTACGTACGTACGTACGT\t*"
            "\tNM:i:5\tAS:i:100");

    CAPTURE(fasta);

    auto results = parse_sam_lines(fasta, "", "");
    REQUIRE(results.size() == 1);

    AlignmentResult alres = results[0];
    CHECK(alres.coverage < 0.8f);
}

TEST_CASE(CUT_TAG " test parse empty field", CUT_TAG) {
    std::string fasta =
            "@SQ\tSN:ref1\tLN:100\n"
            "\t\tref1\t50\t12\t12H20M2I10M1D5H\t\t\t\tACGTACGTACGTACGTACGTACGTACGTACGT\t\tNM:i:"
            "5\tAS:i:"
            "100";
    auto results = parse_sam_lines(fasta, "", "");
    REQUIRE(results.size() == 1);

    AlignmentResult alres = results[0];

    CHECK(std::string("ref1") == alres.genome);
    CHECK(50 == alres.genome_start);
    CHECK(81 == alres.genome_end);
    CHECK(32 == alres.num_events);
    CHECK(12 == alres.strand_start);
    CHECK(44 == alres.strand_end);
    CHECK(2 == alres.num_insertions);
    CHECK(1 == alres.num_deletions);
    CHECK(30 == alres.num_aligned);
    CHECK(28 == alres.num_correct);
    CHECK_THAT(alres.identity, Catch::Matchers::WithinAbs(28.0f / 30, 0.00001f));
    CHECK_THAT(alres.accuracy, Catch::Matchers::WithinAbs(28.0f / (30 + 1 + 2), 0.00001f));
    CHECK(100 == alres.strand_score);
}

TEST_CASE(CUT_TAG " test parse not aligned reference", CUT_TAG) {
    std::string fasta =
            "@SQ\tSN:Lambda\tLN:48400\n@SQ\tSN:Ecoli\tLN:4594032\n1ffb6e83-4d61-44e8-b398-"
            "9eb319a456a7\t4\t*\t0\t0\t*\t*\t0\t0\n";
    auto results = parse_sam_lines(fasta, "", "");
    REQUIRE(results.size() == 1);

    AlignmentResult alres = results[0];

    CHECK(std::string("*") == alres.genome);
    CHECK(0 == alres.genome_start);
    CHECK(0 == alres.genome_end);
    CHECK(0 == alres.num_events);
    CHECK(0 == alres.strand_start);
    CHECK(0 == alres.strand_end);
    CHECK(-1 == alres.num_insertions);
    CHECK(-1 == alres.num_deletions);
    CHECK(-1 == alres.num_aligned);
    CHECK(-1 == alres.num_correct);
    CHECK(-1.f == alres.identity);
    CHECK(-1.f == alres.accuracy);
    CHECK(-1 == alres.strand_score);
}

TEST_CASE(CUT_TAG " test parse with query_seq", CUT_TAG) {
    std::string short_seq = "ACGTACGTACGTACGTACGTAC";
    std::string long_seq = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    std::string long_qstr = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";

    // Test that an empty SAM line seq gets replaced with the query seq in parse_sam_lines
    std::string fasta =
            "@SQ\tSN:ref1\tLN:100\n"
            "read_1\t0\tref1\t50\t12\t2H20M2I10M1D5H\t*\t0\t0\t" +
            std::string("*") + "\t*\tNM:i:5\tAS:i:100";
    auto results = parse_sam_lines(fasta, long_seq, long_qstr);
    REQUIRE(results.size() == 1);

    AlignmentResult alres = results[0];

    CHECK(long_seq == alres.sequence);
    CHECK(long_qstr == alres.qstring);

    // Test that a non empty SAM line seq does not get replaced with the query seq in parse_sam_lines
    fasta = "@SQ\tSN:ref1\tLN:100\n"
            "read_1\t0\tref1\t50\t12\t2H20M2I10M1D5H\t*\t0\t0\t" +
            short_seq + "\t*\tNM:i:5\tAS:i:100";
    results = parse_sam_lines(fasta, long_seq, long_qstr);
    REQUIRE(results.size() == 1);

    alres = results[0];
    CHECK(short_seq == alres.sequence);
    CHECK(std::string("*") == alres.qstring);
}

TEST_CASE(CUT_TAG " test parse with no alignment section does not crash", CUT_TAG) {
    std::string const sam = "@SQ\tSN:Lambda\tLN:48400\n@SQ\tSN:Ecoli\tLN:4594032\n";
    std::string const long_seq = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    std::string const long_qstr = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
    auto alignments = parse_sam_lines(sam, long_seq, long_qstr);
    CHECK(alignments.empty());
}

TEST_CASE(CUT_TAG " test result sorting", CUT_TAG) {
    constexpr int secondary_flag = 256;
    constexpr int supplementary_flag = 2048;
    std::string const long_seq = "ACGTACGTACGTACGTACGTACGTACGTACGT";
    std::string const long_qstr = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";

    // Create the lines in the mock SAM file.
    std::vector<std::string> alignment_sections;
    for (const int flags :
         {0, secondary_flag, supplementary_flag, secondary_flag | supplementary_flag}) {
        const auto flags_str = std::to_string(flags);
        std::string section = "read_";
        section += flags_str;
        section += "\t";
        section += flags_str;
        section += "\n";
        alignment_sections.emplace_back(std::move(section));
    }

    // Go through every permutation of the lines.
    std::sort(alignment_sections.begin(), alignment_sections.end());
    do {
        const auto sam = std::accumulate(alignment_sections.begin(), alignment_sections.end(),
                                         std::string{});
        const auto results = parse_sam_lines(sam, long_seq, long_qstr);
        // 4 flag combos in, 4 results out.
        REQUIRE(results.size() == 4);
        // The first result should be the primary (flags == 0).
        CHECK(results.front().name == "read_0");
    } while (std::next_permutation(alignment_sections.begin(), alignment_sections.end()));
}
