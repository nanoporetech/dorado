#include "../dorado/secondary/features/encoder_read_alignment.h"
#include "TestUtils.h"
#include "hts_utils/fai_utils.h"
#include "local_haplotagging.h"
#include "polish/polish_impl.h"
#include "secondary/common/variant.h"
#include "secondary/features/haplotag_source.h"
#include "utils/container_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

namespace dorado::polisher {
inline bool operator==(const polisher::HaplotagResults& lhs, const polisher::HaplotagResults& rhs) {
    return std::tie(lhs.region_haplotags, lhs.candidate_sites, lhs.merged_pass_variants) ==
           std::tie(rhs.region_haplotags, rhs.candidate_sites, rhs.merged_pass_variants);
}

inline std::ostream& operator<<(std::ostream& os, const polisher::HaplotagResults& a) {
    os << "region_haplotags.size = " << std::size(a.region_haplotags) << '\n';
    for (int64_t i = 0; i < std::ssize(a.region_haplotags); ++i) {
        os << "    [region i = " << i
           << "] region_haplotags[i].size = " << std::size(a.region_haplotags[i]) << '\n';
        for (const auto& [key, val] : a.region_haplotags[i]) {
            os << "{\"" << key << "\", " << val << "},\n";
        }
        os << '\n';
    }
    os << "candidate_sites.size = " << std::size(a.candidate_sites) << '\n';
    int64_t k = 0;
    for (const auto& [key, val] : a.candidate_sites) {
        os << "    [candidate_sites k = " << k << "] key = '" << key << "', val = {"
           << utils::print_container_as_string(val, ", ", false) << "}\n";
        ++k;
    }
    os << "merged_pass_variants.size = " << std::size(a.merged_pass_variants) << '\n';
    for (int64_t i = 0; i < std::ssize(a.merged_pass_variants); ++i) {
        os << "    [i = " << i << "] variant: " << a.merged_pass_variants[i] << '\n';
    }
    return os;
}

}  // namespace dorado::polisher

namespace dorado::polisher::tests {

#define TEST_GROUP "[PolishImpl]"

namespace {
[[maybe_unused]] std::unique_ptr<dorado::secondary::EncoderBase> helper_create_encoder(
        const std::filesystem::path& in_ref_fn,
        const std::filesystem::path& in_bam_aln_fn) {
    const std::vector<std::string> dtypes{};
    const std::string tag_name{};
    const int32_t tag_value{0};
    const bool tag_keep_missing{false};
    const std::string read_group{};
    const int32_t min_mapq{1};
    const int32_t max_reads{100};
    const bool row_per_read{false};
    const bool clip_to_zero{true};
    const bool right_align_insertions{false};
    const bool include_dwells{true};
    const bool include_haplotype_column{true};
    const bool include_snp_qv_column{true};
    const double min_snp_accuracy{0.0};
    const secondary::HaplotagSource hap_source{secondary::HaplotagSource::COMPUTE};
    const std::optional<std::filesystem::path> phasing_bin{};
    const secondary::KadayashiOptions kadayashi_opt{
            .max_clipping = 100000,
            .min_strand_cov = 1,
    };

    std::unique_ptr<secondary::EncoderReadAlignment> encoder =
            std::make_unique<dorado::secondary::EncoderReadAlignment>(
                    in_ref_fn, in_bam_aln_fn, dtypes, tag_name, tag_value, tag_keep_missing,
                    read_group, min_mapq, max_reads, min_snp_accuracy, row_per_read, include_dwells,
                    clip_to_zero, right_align_insertions, include_haplotype_column, hap_source,
                    phasing_bin, include_snp_qv_column, kadayashi_opt);

    return encoder;
}
}  // namespace

CATCH_TEST_CASE("convert_variants", TEST_GROUP) {
    struct TestCase {
        std::string name;
        std::vector<kadayashi::variant_dorado_style_t> kadayashi_variants;
        int32_t seq_id;
        int32_t ploidy;
        float pass_min_qual;
        std::vector<secondary::Variant> expected;
    };

    // clang-format off
    const std::vector<TestCase> test_cases {
        TestCase{"Empty input", {}, 0, 2, 3.0f, {}},
        TestCase{
            "Single hom variant",
            {
                kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T"}, {1, 1}},
            },
            123, 2, 3.0f,
            {
                secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}}, 0, 0},
            },
        },
        TestCase{
            "Single het variant where one alt matches the ref",
            {
                kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T"}, {0, 1} },
            },
            123, 2, 3.0f,
            {
                secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}}, 0, 0},
            },
        },
        TestCase{
            "Single het variant where both alts differ from ref",
            {
                kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {"T", "C"}, {0, 1}},
            },
            123, 2, 3.0f,
            {
                secondary::Variant{123, 5, "A", {"C", "T"}, "PASS", {}, 60.0f, {{"GT", "1/2"}, {"GQ", "60"}}, 0, 0},
            },
        },
        TestCase{
            "Malformed input, there are no alts. Return a '.' as the alt and the filter so this variant can be filtered out later.",
            {
                kadayashi::variant_dorado_style_t{true, true, 5, 60, "A", {}, {0, 1}},
            },
            123, 2, 3.0f,
            {
                secondary::Variant{123, 5, "A", {"."}, ".", {}, 60.0f, {{"GT", "0"}, {"GQ", "60"}}, 0, 0},
            },
        },
        TestCase{
            "Check that is_confident and is_phased play no impact on conversion (they should be ignored). Otherwise, same as the hom test ase above.",
            {
                kadayashi::variant_dorado_style_t{false, false, 5, 60, "A", {"T"}, {1, 1}},
            },
            123, 2, 3.0f,
            {
                secondary::Variant{123, 5, "A", {"T"}, "PASS", {}, 60.0f, {{"GT", "1/1"}, {"GQ", "60"}}, 0, 0},
            },
        },
    };
    // clang-format on

    // Not using Catch2's GENERATE because it explodes on MSVC.
    for (const auto& test_case : test_cases) {
        CATCH_CAPTURE(test_case.name);
        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.name);

        const std::vector<secondary::Variant> result =
                polisher::convert_variants(test_case.kadayashi_variants, test_case.seq_id,
                                           test_case.ploidy, test_case.pass_min_qual);

        CATCH_CHECK(result == test_case.expected);
    }
}

CATCH_TEST_CASE("haplotag_regions_in_parallel", TEST_GROUP) {
    // Test data.
    const std::filesystem::path test_data_dir = get_data_dir("variant") / "test-02-supertiny";
    const std::filesystem::path in_bam_aln_fn = test_data_dir / "in.aln.bam";
    const std::filesystem::path in_ref_fn = test_data_dir / "in.ref.fasta.gz";

    // Static because Catch2 lambda won't capture it.
    const std::vector<std::pair<std::string, int64_t>> loaded_draft_lens =
            utils::load_seq_lengths(in_ref_fn);

    struct TestCase {
        std::string name;
        std::vector<secondary::Window> regions;
        std::vector<std::pair<std::string, int64_t>> draft_lens;
        int32_t num_encoders;
        int32_t num_threads;
        int32_t ploidy;
        float pass_min_qual;
        polisher::HaplotagResults expected;
        bool expect_throw = false;
    };

    // clang-format off
    const std::vector<TestCase> test_cases {
        TestCase{"Empty input, empty output", {}, {}, 1, 2, 2, 3.0f, {}, false},
        TestCase{"Zero encoders, should throw", {}, {}, 0, 2, 2, 3.0f, {}, true},
        TestCase{"Zero threads, should throw", {}, {}, 2, 0, 2, 3.0f, {}, true},
        TestCase{"Zero ploidy, should throw", {}, {}, 2, 2, 0, 3.0f, {}, true},
        TestCase{
                "Normal case",
                {
                    secondary::Window{
                        /*seq_id*/ 0, 10000, /*start*/ 0, /*end*/ 300, 0, 0, -1
                    },
                    secondary::Window{
                        /*seq_id*/ 0, 10000, /*start*/ 1000, /*end*/ 1800, 0, 0, -1
                    },
                    secondary::Window{
                        /*seq_id*/ 0, 10000, /*start*/ 7000, /*end*/ 7500, 0, 0, -1
                    },
                },
                loaded_draft_lens, 2, 2, 2, 3.0f,
                {
                    /*.region_haplotags =*/ {
                        {
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 2},
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 2},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 1},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 2},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 2},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 2},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 2},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 2},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 2},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 2},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 2},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 1},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 2},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 2},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 1},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 2},
                        },
                        {
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 2},
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 2},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 1},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 2},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 2},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 2},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 2},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 2},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 2},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 2},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 2},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 1},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 2},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 2},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 1},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 2},
                        },
                        {
                            {"563ecca1-30dd-4dd9-991a-d417d827c803", 1},
                            {"02551418-20c9-4b4b-9d1b-9bee36342895", 2},
                            {"1e70cda3-c41f-4d19-9c14-94d8d64e619c", 2},
                            {"7d23577c-5c93-4d41-83bd-b652e687deee", 1},
                            {"e0af6c87-8655-4603-97b7-0ad5ba860df2", 1},
                            {"a27cad27-2297-40d4-8666-40a4742eb2ed", 2},
                            {"627ea9e1-5204-4a2c-ae54-1e1be8bbbbe6", 2},
                            {"61ab09d6-072f-4ab2-b14b-b0a1e38a3419", 2},
                            {"c488f4c5-1639-4be1-92f6-948f29b7d822", 2},
                            {"b4139858-e420-4780-94e6-375542c2d2e8", 1},
                            {"ac863a7d-932e-42fa-91c1-7814d7f810f9", 2},
                            {"49b05d0d-97ac-449e-804b-35b35e05ce28", 1},
                            {"de45db56-e704-4524-af88-06a2f98c270e", 2},
                            {"3fdc1b9b-7186-411e-af92-e93a1086754c", 2},
                            {"7b2095d4-08f7-448d-aa9d-55c9568fb49d", 2},
                            {"e7e27cb5-1144-49dd-8ec4-09a75937a091", 1},
                            {"3d7a9813-67be-4b84-b66a-0269aa108340", 2},
                            {"d5560893-59c8-417c-a929-d62b4d19a1ca", 2},
                            {"4fd81aa2-cb77-4994-a8a5-70e6228f255e", 1},
                            {"dbe9785a-fa25-454c-9960-fd65fb99a040", 2},
                        },
                    },
                    /*.candidate_sites =*/ {{"chr20", {1002, 1613, 1619}}},
                    /*.merged_pass_variants =*/ {
                        secondary::Variant{0,   93, "C", {"T"}, {"PASS"}, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}}, 0, 0},
                        secondary::Variant{0, 1471, "T", {"G"}, {"PASS"}, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}}, 0, 0},
                        secondary::Variant{0, 7429, "C", {"T"}, {"PASS"}, {}, 60.0f, {{"GT", "0/1"}, {"GQ", "60"}}, 0, 0},
                    },
                },
                false,
        },
        TestCase{
                "Sequence ID out of bounds for one region, should throw",
                {
                    secondary::Window{
                        /*seq_id*/ 0, 10000, /*start*/ 0, /*end*/ 300, 0, 0, -1
                    },
                    secondary::Window{
                        /*seq_id*/ 0, 10000, /*start*/ 1000, /*end*/ 1800, 0, 0, -1
                    },
                    // There is only one draft sequence, but specified seq_id = 5 here.
                    secondary::Window{
                        /*seq_id*/ 5, 10000, /*start*/ 7000, /*end*/ 7500, 0, 0, -1
                    },
                },
                loaded_draft_lens, 2, 2, 2, 3.0f, {}, true,
        },
    };
    // clang-format on

    // Not using Catch2's GENERATE because it explodes on MSVC.
    for (const auto& test_case : test_cases) {
        CATCH_CAPTURE(test_case.name);
        CATCH_INFO(TEST_GROUP << " Test name: " << test_case.name);

        // Initialize the encoders for this test.
        std::vector<std::unique_ptr<secondary::EncoderBase>> encoders;
        encoders.reserve(test_case.num_encoders);
        for (int32_t i = 0; i < test_case.num_encoders; ++i) {
            encoders.emplace_back(helper_create_encoder(in_ref_fn, in_bam_aln_fn));
        }

        if (test_case.expect_throw) {
            CATCH_CHECK_THROWS(polisher::haplotag_regions_in_parallel(
                    encoders, test_case.regions, test_case.draft_lens, test_case.num_threads,
                    test_case.ploidy, test_case.pass_min_qual));
        } else {
            const polisher::HaplotagResults result = polisher::haplotag_regions_in_parallel(
                    encoders, test_case.regions, test_case.draft_lens, test_case.num_threads,
                    test_case.ploidy, test_case.pass_min_qual);

            CATCH_CHECK(result == test_case.expected);
        }
    }
}

}  // namespace dorado::polisher::tests