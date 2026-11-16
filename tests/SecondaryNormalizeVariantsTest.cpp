#include "secondary/common/variant.h"
#include "secondary/consensus/variant_calling.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <string>

namespace dorado::secondary::consensus::tests {

#define TEST_GROUP "[SecondaryConsensus]"

CATCH_TEST_CASE("collapse_to_haploid", TEST_GROUP) {
    struct TestCase {
        std::string test_name;
        Variant variant;
        Variant expected;
        bool require_hom;
        bool expect_throw = false;
    };

    // clang-format off
    auto [test_case] = GENERATE_REF(table<TestCase>({
        TestCase{
            "Empty test",
            Variant{0, 0, "", {"."}, {Filters::NonVariant}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 0, 0},
            Variant{0, 0, "", {"."}, {Filters::NonVariant}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 0, 0},
            true,
            false,
        },
        TestCase{
            "Haploid, require_hom=true, should be no-op",
            Variant{0, 4, "CAT", {"TGC"}, {Filters::Pass}, {}, 30.0f, {{"GT", "1"}, {"GQ", "30"}}, 3, 7},
            Variant{0, 4, "CAT", {"TGC"}, {Filters::Pass}, {}, 30.0f, {{"GT", "1"}, {"GQ", "30"}}, 3, 7},
            true,
            false,
        },
        TestCase{
            "Haploid, require_hom=false, should be no-op",
            Variant{0, 4, "CAT", {"TGC"}, {Filters::Pass}, {}, 30.0f, {{"GT", "1"}, {"GQ", "30"}}, 3, 7},
            Variant{0, 4, "CAT", {"TGC"}, {Filters::Pass}, {}, 30.0f, {{"GT", "1"}, {"GQ", "30"}}, 3, 7},
            false,
            false,
        },
        TestCase{
            "Diploid gVCF N reference record should collapse to haploid no-call",
            Variant{0, 4, "N", {"<*>"}, {Filters::NonVariant}, {{"END", "5"}}, 0.0f, {{"GT", "./."}, {"GQ", "."}, {"LEN", "1"}}, 4, 5},
            Variant{0, 4, "N", {"<*>"}, {Filters::NonVariant}, {{"END", "5"}}, -1.0f, {{"GT", "."}, {"GQ", "."}, {"LEN", "1"}}, 4, 5},
            true,
            false,
        },
        TestCase{
            "Diploid, homozygous, require_hom=true, should always be collapsed",
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1/1"}, {"GQ", "43"}}, 2, 4},
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1"}, {"GQ", "43"}}, 2, 4},
            true,
            false,
        },
        TestCase{
            "Diploid, homozygous, require_hom=false, should always be collapsed",
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1/1"}, {"GQ", "43"}}, 2, 4},
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1"}, {"GQ", "43"}}, 2, 4},
            false,
            false,
        },
        TestCase{
            "Diploid 2, heterozygous, require_hom=true, should be filtered",
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "0/1"}, {"GQ", "44"}}, 4, 8},
            Variant{0, 4, "AAA", {"A"}, {Filters::IncompatZyg}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 4, 8},
            true,
            false,
        },
        TestCase{
            "Diploid 2, heterozygous, require_hom=false, should be collapsed",
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "0/1"}, {"GQ", "44"}}, 4, 8},
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "1"}, {"GQ", "44"}}, 4, 8},
            false,
            false,
        },
         TestCase{
            "Diploid 2, hetalt, should be filtered",
            Variant{0, 7, "AA", {"AC", "A"}, {Filters::Pass}, {}, 45.0f, {{"GT", "1/2"}, {"GQ", "45"}}, 8, 10},
            Variant{0, 7, "AA", {"AC", "A"}, {Filters::IncompatZyg}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 8, 10},
            true,
            false,
        },
        TestCase{
            "Triploid, homozygous, require_hom=true, should always be collapsed",
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1/1/1"}, {"GQ", "43"}}, 2, 4},
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1"}, {"GQ", "43"}}, 2, 4},
            true,
            false,
        },
        TestCase{
            "Triploid, homozygous, require_hom=false, should always be collapsed",
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1/1/1"}, {"GQ", "43"}}, 2, 4},
            Variant{0, 2, "CA", {"C"}, {Filters::Pass}, {}, 43.0f, {{"GT", "1"}, {"GQ", "43"}}, 2, 4},
            false,
            false,
        },
        TestCase{
            "Triploid 2, heterozygous, require_hom=true, should be filtered",
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "0/0/1"}, {"GQ", "44"}}, 4, 8},
            Variant{0, 4, "AAA", {"A"}, {Filters::IncompatZyg}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 4, 8},
            true,
            false,
        },
        TestCase{
            "Triploid 2, heterozygous, require_hom=false, should be collapsed",
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "0/0/1"}, {"GQ", "44"}}, 4, 8},
            Variant{0, 4, "AAA", {"A"}, {Filters::Pass}, {}, 44.0f, {{"GT", "1"}, {"GQ", "44"}}, 4, 8},
            false,
            false,
        },
        TestCase{
            "Triploid 2, hetalt, should be filtered",
            Variant{0, 7, "AA", {"AC", "A"}, {Filters::Pass}, {}, 45.0f, {{"GT", "0/1/2"}, {"GQ", "45"}}, 8, 10},
            Variant{0, 7, "AA", {"AC", "A"}, {Filters::IncompatZyg}, {}, 0.0f, {{"GT", "0"}, {"GQ", "0"}}, 8, 10},
            true,
            false,
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    if (test_case.expect_throw) {
        CATCH_CHECK_THROWS(collapse_to_haploid(test_case.variant, test_case.require_hom));
    } else {
        const Variant result = collapse_to_haploid(test_case.variant, test_case.require_hom);
        CATCH_CHECK(test_case.expected == result);
    }
}

CATCH_TEST_CASE("update_filters handles PASS and no-variant filters", TEST_GROUP) {
    const auto check_update = [](Filters filters, Filters::Filter new_filter, Filters expected,
                                 const std::string_view message) {
        filters.update(new_filter);
        CATCH_INFO(message);
        CATCH_CHECK(filters == expected);
    };

    check_update({Filters::NonVariant}, Filters::LowQual, {Filters::LowQual},
                 "update with a filter overrides non-variant");
    check_update({Filters::LowQual}, Filters::IncompatZyg,
                 {Filters::LowQual | Filters::IncompatZyg}, "update merges filters");
    check_update({Filters::LowQual}, Filters::LowQual, {Filters::LowQual},
                 "update with existing filter is no-op");
    check_update({Filters::Missing}, Filters::Pass, {Filters::Pass}, "update from missing");
    check_update({Filters::LowQual}, Filters::Pass, {Filters::Pass}, "update with pass overrides");
    check_update({Filters::Pass}, Filters::LowQual, {Filters::LowQual},
                 "update of existing pass removes pass");
    check_update({Filters::NonVariant}, Filters::LowQual, {Filters::LowQual},
                 "update of existing non-variant removes non-variant");
    check_update({Filters::LowQual}, Filters::NonVariant, {Filters::NonVariant},
                 "update with non-variant overrides");
    check_update({Filters::LowQual}, Filters::Missing, {Filters::LowQual},
                 "update with missing is no-op");
}

}  // namespace dorado::secondary::consensus::tests
