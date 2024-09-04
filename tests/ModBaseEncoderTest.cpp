#include "modbase/ModbaseEncoder.h"

#include "modbase/encode_kmer.h"
#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define TEST_GROUP "[modbase_encoder]"

using namespace dorado::modbase;

namespace {

// Base to u32 one-hot encoding
std::unordered_map<char, uint32_t> encode_map = {{
        {'N', uint32_t{0}},
        {'A', uint32_t{1} << (0 << 3)},
        {'C', uint32_t{1} << (1 << 3)},
        {'G', uint32_t{1} << (2 << 3)},
        {'T', uint32_t{1} << (3 << 3)},
}};

// Sequence int to base char
std::unordered_map<int8_t, char> seq_int_map = {{
        {0, 'A'},
        {1, 'C'},
        {2, 'G'},
        {3, 'T'},
}};

// Takes a string of bases and converts into the one hot encoding for testing
std::vector<int8_t> encode_bases(const std::string& bases) {
    std::vector<int8_t> out(bases.size() * 4);
    size_t count_bases = 0;
    int8_t* p_out = &out[0];
    for (size_t i = 0; i < bases.size(); ++i) {
        auto& base = bases[i];
        if (base == ' ') {
            continue;
        }
        if (encode_map.find(base) == encode_map.end()) {
            throw std::runtime_error("Cannot encode unknown sequence base: '" +
                                     std::to_string(base) + "'.");
        }
        count_bases++;
        std::memcpy(p_out, &encode_map[base], sizeof(uint32_t));
        p_out += sizeof(uint32_t);
    }
    out.resize(count_bases * 4);
    return out;
}

// Takes an encoded kmer and converts it to a string spaced by the kmer_len for readability
std::string decode_bases(const std::vector<int8_t>& encoded_bases, size_t kmer_len) {
    if (encoded_bases.size() % 4 != 0) {
        throw std::runtime_error("encoded_bases vector must be modulo 4");
    }
    size_t num_bases = encoded_bases.size() / 4;
    std::string s;
    size_t len = kmer_len == 0 ? num_bases : num_bases + num_bases / kmer_len;
    s.reserve(len);
    for (size_t b = 0; b < num_bases; ++b) {
        for (int8_t i = 0; i < 5; ++i) {
            if (i == 4) {
                s += 'N';
                break;
            }
            if (encoded_bases[b * 4 + i] == 1) {
                s += seq_int_map[i];
                break;
            }
        }
        if (kmer_len > 0 && b % kmer_len == kmer_len - 1 && b < num_bases - 1) {
            s += ' ';
        }
    }
    return s;
}

}  // namespace

TEST_CASE("Test kmer encoder helper", TEST_GROUP) {
    SECTION("Happy path") {
        auto [sequence, encoded] = GENERATE(table<std::string, std::vector<int8_t>>(
                {{"N", {0, 0, 0, 0}},
                 {"A", {1, 0, 0, 0}},
                 {"C", {0, 1, 0, 0}},
                 {"G", {0, 0, 1, 0}},
                 {"T", {0, 0, 0, 1}},
                 {"AA", {1, 0, 0, 0, 1, 0, 0, 0}},
                 {"ACGT", {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
                 {"NNN", std::vector<int8_t>(4 * 3, 0)},
                 {"TTAGCT",
                  {0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1}}}));
        CAPTURE(sequence);
        CHECK(encoded == encode_bases(sequence));
    }
    SECTION("Invalid input") {
        auto bad_sequence = GENERATE("X", "a", "AC  x  GT");
        CAPTURE(bad_sequence);
        CHECK_THROWS(encode_bases(bad_sequence));
    }
}

TEST_CASE("Test kmer decoder helper", TEST_GROUP) {
    SECTION("Happy path") {
        auto [sequence, kmer_len, encoded] = GENERATE(table<std::string, int, std::vector<int8_t>>(
                {{"N", 0, {0, 0, 0, 0}},
                 {"A", 0, {1, 0, 0, 0}},
                 {"C", 0, {0, 1, 0, 0}},
                 {"G", 0, {0, 0, 1, 0}},
                 {"T", 0, {0, 0, 0, 1}},
                 {"AA", 0, {1, 0, 0, 0, 1, 0, 0, 0}},
                 {"ACGT", 0, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
                 {"AC GT", 2, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
                 {"A C G T", 1, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
                 {"TT AG CT", 2, {0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
                                  0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1}}}));
        CAPTURE(sequence, kmer_len);
        CHECK(sequence == decode_bases(encoded, kmer_len));
    }
    SECTION("Invalid inputs caught") {
        auto [bad_input] = GENERATE(table<std::vector<int8_t>>({
                {{0}},
                {{0, 0, 0, 0, 0}},
        }));
        CAPTURE(bad_input);
        CHECK_THROWS(decode_bases(bad_input, 1));
    }
}

TEST_CASE("Encode sequence for modified basecalling", TEST_GROUP) {
    const size_t BLOCK_STRIDE = 2;
    const size_t SLICE_BLOCKS = 6;
    std::string sequence{"TATTCAGTAC"};
    auto seq_ints = dorado::utils::sequence_to_ints(sequence);
    //                         T  A     T        T  C     A     G        T     A  C
    std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
    auto seq_to_sig_map = dorado::utils::moves_to_map(moves, BLOCK_STRIDE,
                                                      moves.size() * BLOCK_STRIDE, std::nullopt);

    dorado::modbase::ModBaseEncoder encoder(BLOCK_STRIDE, SLICE_BLOCKS * BLOCK_STRIDE, 1, 1);
    encoder.init(seq_ints, seq_to_sig_map);

    auto slice0 = encoder.get_context(0);  // The T in the NTA 3mer.
    CHECK(slice0.first_sample == 0);
    CHECK(slice0.num_samples == 7);
    CHECK(slice0.lead_samples_needed == 5);
    CHECK(slice0.tail_samples_needed == 0);

    // clang-format off
    std::vector<int8_t> expected_slice0 = {
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, // NTA
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, // TAT
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
    };

    // clang-format on
    CHECK(expected_slice0 == slice0.data);

    std::string bases0 = "NTA NTA NTA NTA NTA NTA NTA TAT TAT TAT TAT ATT";
    CHECK(expected_slice0 == encode_bases(bases0));

    auto slice1 = encoder.get_context(4);  // The C in the TCA 3mer.
    CHECK(slice1.first_sample == 10);
    CHECK(slice1.num_samples == 12);
    CHECK(slice1.lead_samples_needed == 0);
    CHECK(slice1.tail_samples_needed == 0);

    // clang-format off
    std::vector<int8_t> expected_slice1 = {
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
        1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, // ATT
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, // TTC
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, // TTC
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, // TCA
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, // CAG
    };

    // clang-format on
    CHECK(expected_slice1 == slice1.data);
    std::string bases1 = "ATT ATT TTC TTC TCA TCA TCA TCA CAG CAG CAG CAG";
    CHECK(expected_slice1 == encode_bases(bases1));

    auto slice2 = encoder.get_context(9);  // The C in the ACN 3mer.
    CHECK(slice2.first_sample == 31);
    CHECK(slice2.num_samples == 9);
    CHECK(slice2.lead_samples_needed == 0);
    CHECK(slice2.tail_samples_needed == 3);

    // clang-format off
    std::vector<int8_t> expected_slice2 = {
        0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, // GTA
        0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, // TAC
        0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, // TAC
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, // ACN
    };
    // clang-format on
    CHECK(expected_slice2 == slice2.data);

    std::string bases2 = "GTA TAC TAC ACN ACN ACN ACN ACN ACN ACN ACN ACN";
    CHECK(expected_slice2 == encode_bases(bases2));
    CHECK(bases2.size() == decode_bases(expected_slice2, 3).size());
    CHECK(bases2 == decode_bases(expected_slice2, 3));
}

TEST_CASE("Encode kmer for chunk mods models - stride 2", TEST_GROUP) {
    const size_t BLOCK_STRIDE = 2;
    std::string sequence{"TATTCAGTAC"};
    auto seq_ints = dorado::utils::sequence_to_ints(sequence);
    //                         T  A     T        T  C     A     G        T     A  C
    std::vector<uint8_t> moves{1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0};
    auto seq_to_sig_map = dorado::utils::moves_to_map(moves, BLOCK_STRIDE,
                                                      moves.size() * BLOCK_STRIDE, std::nullopt);

    const size_t whole_context = moves.size() * BLOCK_STRIDE;

    auto test_chunk_enc = [&](size_t kmer_len, size_t padding, bool is_centered,
                              const std::string& expected_bases) {
        auto result = encode_kmer_chunk(seq_ints, seq_to_sig_map, kmer_len, whole_context, padding,
                                        is_centered);

        CAPTURE(seq_ints, seq_to_sig_map, whole_context);
        CAPTURE(kmer_len, padding, is_centered);
        CHECK(result.size() == (whole_context + 2 * padding) * 4 * kmer_len);
        CHECK(decode_bases(result, kmer_len) == expected_bases);
    };

    SECTION("1mer not centered") {
        const size_t kmer_len = 1, padding = 0;
        const bool is_centered = false;
        std::string kmer_bases =
                "T T A A A A T T T T T T T T C C C C A A A A G G G G G G T T T T A A C C C C C C";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer not centered") {
        const size_t kmer_len = 3, padding = 0;
        const bool is_centered = false;
        std::string kmer_bases =
                //T  T   A   A   A   A   T   T
                "TAT TAT ATT ATT ATT ATT TTC TTC "
                //T  T   T   T   T   T   C   C
                "TTC TTC TTC TTC TCA TCA CAG CAG "
                //C  C   A   A   A   A   G   G
                "CAG CAG AGT AGT AGT AGT GTA GTA "
                //G  G   G   G   T   T   T   T
                "GTA GTA GTA GTA TAC TAC TAC TAC "
                //A  A   C   C   C   C   C   C
                "ACN ACN CNN CNN CNN CNN CNN CNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer not centered with padding") {
        const size_t kmer_len = 3, padding = 3;
        const bool is_centered = false;
        std::string kmer_bases =
                "NNN NNN NNN "  // +3 3-mers of padding each end
                //T  T   A   A   A   A   T   T
                "TAT TAT ATT ATT ATT ATT TTC TTC "
                //T  T   T   T   T   T   C   C
                "TTC TTC TTC TTC TCA TCA CAG CAG "
                //C  C   A   A   A   A   G   G
                "CAG CAG AGT AGT AGT AGT GTA GTA "
                //G  G   G   G   T   T   T   T
                "GTA GTA GTA GTA TAC TAC TAC TAC "
                //A  A   C   C   C   C   C   C
                "ACN ACN CNN CNN CNN CNN CNN CNN "
                "NNN NNN NNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer centered") {
        const size_t kmer_len = 3, padding = 0;
        const bool is_centered = true;
        std::string kmer_bases =
                //T   T   A   A   A   A   T   T
                "NTA NTA TAT TAT TAT TAT ATT ATT "
                //T   T   T   T   T   T   C   C
                "ATT ATT ATT ATT TTC TTC TCA TCA "
                //C   C   A   A   A   A   G   G
                "TCA TCA CAG CAG CAG CAG AGT AGT "
                //G   G   G   G   T   T   T   T
                "AGT AGT AGT AGT GTA GTA GTA GTA "
                //A   A   C   C   C   C   C   C
                "TAC TAC ACN ACN ACN ACN ACN ACN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer centered with padding") {
        const size_t kmer_len = 3, padding = 1;
        const bool is_centered = true;
        std::string kmer_bases =
                "NNN "  // +1 3-mer of padding each end
                //T  T   A  A   A  A   T  T
                "NTA NTA TAT TAT TAT TAT ATT ATT "
                //T   T   T   T   T   T   C   C
                "ATT ATT ATT ATT TTC TTC TCA TCA "
                //C   C   A   A   A   A   G   G
                "TCA TCA CAG CAG CAG CAG AGT AGT "
                //G   G   G   G   T   T   T   T
                "AGT AGT AGT AGT GTA GTA GTA GTA "
                //A   A   C   C   C   C   C   C
                "TAC TAC ACN ACN ACN ACN ACN ACN "
                "NNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("5-mer centered") {
        const size_t kmer_len = 5, padding = 0;
        const bool is_centered = true;
        std::string kmer_bases =
                // T     T     A     A     A     A     T     T
                "NNTAT NNTAT NTATT NTATT NTATT NTATT TATTC TATTC "
                // T     T     T     T     T     T     C     C
                "TATTC TATTC TATTC TATTC ATTCA ATTCA TTCAG TTCAG "
                // C     C     A     A     A     A     G     G
                "TTCAG TTCAG TCAGT TCAGT TCAGT TCAGT CAGTA CAGTA "
                // G     G     G     G     T     T     T     T
                "CAGTA CAGTA CAGTA CAGTA AGTAC AGTAC AGTAC AGTAC "
                // A     A     C     C     C     C     C     C
                "GTACN GTACN TACNN TACNN TACNN TACNN TACNN TACNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("5-mer centered with padding") {
        const size_t kmer_len = 5, padding = 2;
        const bool is_centered = true;
        std::string kmer_bases =
                "NNNNN NNNNN "  // +2 5-mer of padding each end
                // T     T     A     A     A     A     T     T
                "NNTAT NNTAT NTATT NTATT NTATT NTATT TATTC TATTC "
                // T     T     T     T     T     T     C     C
                "TATTC TATTC TATTC TATTC ATTCA ATTCA TTCAG TTCAG "
                // C     C     A     A     A     A     G     G
                "TTCAG TTCAG TCAGT TCAGT TCAGT TCAGT CAGTA CAGTA "
                // G     G     G     G     T     T     T     T
                "CAGTA CAGTA CAGTA CAGTA AGTAC AGTAC AGTAC AGTAC "
                // A     A     C     C     C     C     C     C
                "GTACN GTACN TACNN TACNN TACNN TACNN TACNN TACNN "
                "NNNNN NNNNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("9-mer not centered") {
        // Although this seems excessive, the 9-mer has a SIMD fast-path for AVX platforms
        const size_t kmer_len = 9, padding = 0;
        const bool is_centered = false;
        std::string kmer_bases =
                "TATTCAGTA TATTCAGTA "  // TT
                "ATTCAGTAC ATTCAGTAC "  // AA
                "ATTCAGTAC ATTCAGTAC "  // AA
                "TTCAGTACN TTCAGTACN "  // TT
                "TTCAGTACN TTCAGTACN "  // TT
                "TTCAGTACN TTCAGTACN "  // TT
                "TCAGTACNN TCAGTACNN "  // TT
                "CAGTACNNN CAGTACNNN "  // CC
                "CAGTACNNN CAGTACNNN "  // CC
                "AGTACNNNN AGTACNNNN "  // AA
                "AGTACNNNN AGTACNNNN "  // AA
                "GTACNNNNN GTACNNNNN "  // GG
                "GTACNNNNN GTACNNNNN "  // GG
                "GTACNNNNN GTACNNNNN "  // GG
                "TACNNNNNN TACNNNNNN "  // TT
                "TACNNNNNN TACNNNNNN "  // TT
                "ACNNNNNNN ACNNNNNNN "  // AA
                "CNNNNNNNN CNNNNNNNN "  // CC
                "CNNNNNNNN CNNNNNNNN "  // CC
                "CNNNNNNNN CNNNNNNNN";  // CC
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("9-mer not centered with padding") {
        // Although this seems excessive, the 9-mer has a SIMD fast-path for AVX platforms
        const size_t kmer_len = 9, padding = 2;
        const bool is_centered = false;
        std::string kmer_bases =
                "NNNNNNNNN NNNNNNNNN "  // TT
                "TATTCAGTA TATTCAGTA "  // TT
                "ATTCAGTAC ATTCAGTAC "  // AA
                "ATTCAGTAC ATTCAGTAC "  // AA
                "TTCAGTACN TTCAGTACN "  // TT
                "TTCAGTACN TTCAGTACN "  // TT
                "TTCAGTACN TTCAGTACN "  // TT
                "TCAGTACNN TCAGTACNN "  // TT
                "CAGTACNNN CAGTACNNN "  // CC
                "CAGTACNNN CAGTACNNN "  // CC
                "AGTACNNNN AGTACNNNN "  // AA
                "AGTACNNNN AGTACNNNN "  // AA
                "GTACNNNNN GTACNNNNN "  // GG
                "GTACNNNNN GTACNNNNN "  // GG
                "GTACNNNNN GTACNNNNN "  // GG
                "TACNNNNNN TACNNNNNN "  // TT
                "TACNNNNNN TACNNNNNN "  // TT
                "ACNNNNNNN ACNNNNNNN "  // AA
                "CNNNNNNNN CNNNNNNNN "  // CC
                "CNNNNNNNN CNNNNNNNN "  // CC
                "CNNNNNNNN CNNNNNNNN "  // CC
                "NNNNNNNNN NNNNNNNNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("9-mer centered") {
        const size_t kmer_len = 9, padding = 0;
        const bool is_centered = true;
        std::string kmer_bases =
                "NNNNTATTC NNNNTATTC "  // TT
                "NNNTATTCA NNNTATTCA "  // AA
                "NNNTATTCA NNNTATTCA "  // AA
                "NNTATTCAG NNTATTCAG "  // TT
                "NNTATTCAG NNTATTCAG "  // TT
                "NNTATTCAG NNTATTCAG "  // TT
                "NTATTCAGT NTATTCAGT "  // TT
                "TATTCAGTA TATTCAGTA "  // CC
                "TATTCAGTA TATTCAGTA "  // CC
                "ATTCAGTAC ATTCAGTAC "  // AA
                "ATTCAGTAC ATTCAGTAC "  // AA
                "TTCAGTACN TTCAGTACN "  // GG
                "TTCAGTACN TTCAGTACN "  // GG
                "TTCAGTACN TTCAGTACN "  // GG
                "TCAGTACNN TCAGTACNN "  // TT
                "TCAGTACNN TCAGTACNN "  // TT
                "CAGTACNNN CAGTACNNN "  // AA
                "AGTACNNNN AGTACNNNN "  // CC
                "AGTACNNNN AGTACNNNN "  // CC
                "AGTACNNNN AGTACNNNN";  // CC
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }
}

TEST_CASE("Encode kmer for chuck mods models - stride 5", TEST_GROUP) {
    const size_t BLOCK_STRIDE = 5;
    std::string sequence{"TAGTCA"};
    auto seq_ints = dorado::utils::sequence_to_ints(sequence);
    //                         T        A     G  T  C     A
    std::vector<uint8_t> moves{1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0};
    auto seq_to_sig_map = dorado::utils::moves_to_map(moves, BLOCK_STRIDE,
                                                      moves.size() * BLOCK_STRIDE, std::nullopt);

    const size_t whole_context = moves.size() * BLOCK_STRIDE;

    auto test_chunk_enc = [&](size_t kmer_len, size_t padding, bool is_centered,
                              const std::string& expected_bases) {
        auto result = encode_kmer_chunk(seq_ints, seq_to_sig_map, kmer_len, whole_context, padding,
                                        is_centered);

        CAPTURE(seq_ints, seq_to_sig_map, whole_context);
        CAPTURE(kmer_len, padding, is_centered);
        CHECK(result.size() == (whole_context + 2 * padding) * 4 * kmer_len);
        CHECK(decode_bases(result, kmer_len) == expected_bases);
    };

    SECTION("1mer not centered") {
        const size_t kmer_len = 1, padding = 0;
        const bool is_centered = false;
        std::string kmer_bases =
                "T T T T T T T T T T T T T T T "
                "A A A A A A A A A A "
                "G G G G G "
                "T T T T T "
                "C C C C C C C C C C "
                "A A A A A A A A A A";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer not centered") {
        const size_t kmer_len = 3, padding = 0;
        const bool is_centered = false;
        std::string kmer_bases =
                "TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG "
                "AGT AGT AGT AGT AGT AGT AGT AGT AGT AGT "
                "GTC GTC GTC GTC GTC "
                "TCA TCA TCA TCA TCA "
                "CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN "
                "ANN ANN ANN ANN ANN ANN ANN ANN ANN ANN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer not centered with padding") {
        const size_t kmer_len = 3, padding = 3;
        const bool is_centered = false;
        std::string kmer_bases =
                "NNN NNN NNN "
                "TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG "
                "AGT AGT AGT AGT AGT AGT AGT AGT AGT AGT "
                "GTC GTC GTC GTC GTC "
                "TCA TCA TCA TCA TCA "
                "CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN "
                "ANN ANN ANN ANN ANN ANN ANN ANN ANN ANN "
                "NNN NNN NNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }

    SECTION("3-mer centered with padding") {
        const size_t kmer_len = 3, padding = 5;
        const bool is_centered = true;
        std::string kmer_bases =
                "NNN NNN NNN NNN NNN "
                "NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA NTA "
                "TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG "
                "AGT AGT AGT AGT AGT "
                "GTC GTC GTC GTC GTC "
                "TCA TCA TCA TCA TCA TCA TCA TCA TCA TCA "
                "CAN CAN CAN CAN CAN CAN CAN CAN CAN CAN "
                "NNN NNN NNN NNN NNN";
        test_chunk_enc(kmer_len, padding, is_centered, kmer_bases);
    }
}
