#include "modbase/ModBaseContext.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

CATCH_TEST_CASE("basemod context helper encoding", "[bam]") {
    dorado::modbase::ModBaseContext context_handler;
    std::string expected_context = "_:XG:_:AXA";  // C has context CG, and T has context ATA

    context_handler.set_context("A", 0);
    context_handler.set_context("CG", 0);
    context_handler.set_context("G", 0);
    context_handler.set_context("ATA", 1);
    CATCH_CHECK(context_handler.encode() == expected_context);

    std::string sequence = "CTAGACGTTCGACATATTGA";
    std::vector<bool> expected_mask = {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    auto mask = context_handler.get_sequence_mask(sequence);
    CATCH_CHECK(mask == expected_mask);
}

CATCH_TEST_CASE("basemod context helper", "[bam]") {
    dorado::modbase::ModBaseContext context_handler;
    std::string context = "_:XG:_:AXA";  // C has context CG, and T has context ATA
    bool result = context_handler.decode(context, true);
    CATCH_REQUIRE(result);
    CATCH_REQUIRE(context_handler.motif('A').empty());
    CATCH_REQUIRE(context_handler.motif('C') == "CG");
    CATCH_REQUIRE(context_handler.motif('G').empty());
    CATCH_REQUIRE(context_handler.motif('T') == "ATA");
    CATCH_REQUIRE(context_handler.motif_offset('A') == 0);
    CATCH_REQUIRE(context_handler.motif_offset('C') == 0);
    CATCH_REQUIRE(context_handler.motif_offset('G') == 0);
    CATCH_REQUIRE(context_handler.motif_offset('T') == 1);
    auto reencoded = context_handler.encode();
    CATCH_REQUIRE(reencoded == context);
    std::string sequence = "CTAGACGTTCGACATATTGA";
    std::vector<bool> expected_mask = {0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
    auto mask = context_handler.get_sequence_mask(sequence);
    CATCH_REQUIRE(mask == expected_mask);
    uint8_t threshold = 100;
    std::vector<std::string> alphabet = {"A", "C", "m", "G", "x", "T", "y"};

    std::vector<uint8_t> modbase_probs = {
            0,   255, 0,   0,   0,   0,   0,    // C
            0,   0,   0,   0,   0,   255, 0,    // T
            255, 0,   0,   0,   0,   0,   0,    // A
            0,   0,   0,   80,  175, 0,   0,    // Gx
            255, 0,   0,   0,   0,   0,   0,    // A
            0,   230, 25,  0,   0,   0,   0,    // Cm (weak call)
            0,   0,   0,   200, 55,  0,   0,    // Gx (weak call)
            0,   0,   0,   0,   0,   255, 0,    // T
            0,   0,   0,   0,   0,   255, 0,    // T
            0,   30,  225, 0,   0,   0,   0,    // Cm
            0,   0,   0,   255, 0,   0,   0,    // G
            255, 0,   0,   0,   0,   0,   0,    // A
            0,   255, 0,   0,   0,   0,   0,    // C
            255, 0,   0,   0,   0,   0,   0,    // A
            0,   0,   0,   0,   0,   55,  200,  // Ty
            255, 0,   0,   0,   0,   0,   0,    // A
            0,   0,   0,   0,   0,   255, 0,    // T
            0,   0,   0,   0,   0,   255, 0,    // T
            0,   0,   0,   100, 155, 0,   0,    // Gx
            255, 0,   0,   0,   0,   0,   0,    // A
    };
    std::vector<bool> full_mask = {0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0};
    context_handler.update_mask(mask, sequence, alphabet, modbase_probs, threshold);
    CATCH_REQUIRE(mask == full_mask);

    // Test again with a mask where the contextless Gx modification is set to true for all Gs.
    mask = context_handler.get_sequence_mask(sequence);
    for (size_t seq_pos = 0; seq_pos < sequence.size(); ++seq_pos) {
        if (sequence[seq_pos] == 'G') {
            mask[seq_pos] = true;
        }
    }
    context_handler.update_mask(mask, sequence, alphabet, modbase_probs, threshold);
    CATCH_REQUIRE(mask == full_mask);
}
