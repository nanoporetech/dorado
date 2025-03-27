#include "correct/windows.h"
#include "secondary/interval.h"
#include "utils/cigar.h"
#include "utils/overlap.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#define TEST_GROUP "[Correction-Windows-split_alignment]"

// #define DEBUG_CORRECTION_WINDOW_TESTS

#ifdef DEBUG_CORRECTION_WINDOW_TESTS
#include <iostream>
#endif

namespace dorado::correction::window_tests {

CATCH_TEST_CASE("All tests", TEST_GROUP) {
    using namespace dorado::secondary;

    struct TestCase {
        std::string test_name;
        int32_t aln_id = 0;
        utils::Overlap overlap;
        std::string cigar;
        std::vector<Interval> window_intervals;
        bool custom_initial_point = false;
        CigarPoint cigar_point;
        std::vector<std::pair<int32_t, OverlapWindow>> expected;
    };

    // clang-format off
    auto [test_case] = GENERATE(table<TestCase>({
        TestCase{
            "Empty CIGAR",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "",
            {
                Interval{0, 10000},
            },
            false, {},
            {       // Expected results.
            },
        },
        TestCase{
            "Empty interval list",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
            },
            false, {},
            {       // Expected results.
            },
        },
        TestCase{
            "Alignment of 1bp",
            18, utils::Overlap{0, 1, 10000, 5000, 5001, 10000, true}, "1=",
            {
                Interval{0, 10000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 0, 10000, 5000, 5001, 0, 1, 0, 0, 1, 0, 0.0, 1}},
            },
        },
        TestCase{
            "Full length alignment, 1 window, 1 CIGAR event",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
                Interval{0, 10000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 0, 10000, 0, 10000, 0, 10000, 0, 0, 1, 0, 0.0, 10000}},
            },
        },
        TestCase{
            "Full length alignment, 3 windows, 1 CIGAR event",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
                Interval{0, 2000},
                Interval{2000, 7000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 0, 2000, 0, 2000, 0, 2000, 0, 0, 0, 2000, 0.0, 2000}},
                {1, {18, 2000, 7000, 2000, 7000, 2000, 7000, 0, 2000, 0, 7000, 0.0, 5000}},
                {2, {18, 7000, 10000, 7000, 10000, 7000, 10000, 0, 7000, 1, 0, 0.0, 3000}},
            },
        },
        TestCase{
            "Full length alignment, 1 window at the very start of the alignment",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
                Interval{0, 2000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 0, 2000, 0, 2000, 0, 2000, 0, 0, 0, 2000, 0.0, 2000}},
            },
        },
        TestCase{
            "Full length alignment, 1 window in the middle of the alignment",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
                Interval{2000, 7000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 2000, 7000, 2000, 7000, 2000, 7000, 0, 2000, 0, 7000, 0.0, 5000}},
            },
        },
        TestCase{
            "Full length alignment, 1 window at the very end of the alignment",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "10000=",
            {
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 7000, 10000, 7000, 10000, 7000, 10000, 0, 7000, 1, 0, 0.0, 3000}},
            },
        },
        TestCase{
            "Full length alignment, 3 windows, more complicated CIGAR",
            18, utils::Overlap{0, 10000, 10000, 0, 10000, 10000, true}, "300=1I10=2D3000=10I3688=100D2900=",
            {
                Interval{0, 2000},
                Interval{2000, 7000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {0, {18, 0, 2000, 0, 2000, 0, 1999, 0, 0, 4, 1688, 0.0, 2001}},             // 300=1I10=2D1688=
                {1, {18, 2000, 7000, 2000, 7000, 1999, 7009, 4, 1688, 7, 0, 0.0, 5010}},    // 1312=10I3688=
                {2, {18, 7000, 10000, 7000, 10000, 7009, 9909, 7, 0, 9, 0, 0.0, 3000}},     // 100D2900=
            },
        },
        TestCase{
            "Alignment begins within one of the windows",
            18, utils::Overlap{0, 1000, 1000, 3000, 4000, 10000, true}, "1000=",
            {
                Interval{0, 2000},
                Interval{2000, 7000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {1, {18, 2000, 7000, 3000, 4000, 0, 1000, 0, 0, 1, 0, 0.0, 1000}},
            },
        },
        TestCase{
            "Alignment begins in the middle of one window and ends in the middle of the next one",
            18, utils::Overlap{0, 1000, 1000, 6500, 7500, 10000, true}, "1000=",
            {
                Interval{0, 2000},
                Interval{2000, 7000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {1, {18, 2000, 7000, 6500, 7000, 0, 500, 0, 0, 0, 500, 0.0, 500}},
                {2, {18, 7000, 10000, 7000, 7500, 500, 1000, 0, 500, 1, 0, 0.0, 500}},
            },
        },
        TestCase{
            "Alignment overlaps the last window end to end",
            18, utils::Overlap{0, 3000, 3000, 7000, 10000, 10000, true}, "3000=",
            {
                Interval{0, 2000},
                Interval{2000, 7000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {2, {18, 7000, 10000, 7000, 10000, 0, 3000, 0, 0, 1, 0, 0.0, 3000}},
            },
        },
        TestCase{
            "Alignment begins outside of the requested window. Empty output is expected.",
            18, utils::Overlap{0, 1000, 1000, 3000, 4000, 10000, true}, "1000=",
            {
                Interval{0, 2000},
            },
            false, {},
            {       // Expected results.
            },
        },
        TestCase{
            "Alignment begins in a region not covered by windows, and then extends in a region covered by a window",
            18, utils::Overlap{0, 1000, 1000, 6500, 7500, 10000, true}, "1000=",
            {
                Interval{0, 2000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
                {1, {18, 7000, 10000, 7000, 7500, 500, 1000, 0, 500, 1, 0, 0.0, 500}},
            },
        },
        TestCase{
            "Alignment is in between windows. No output should be generated.",
            18, utils::Overlap{0, 1000, 1000, 5000, 6000, 10000, true}, "1000=",
            {
                Interval{0, 2000},
                Interval{7000, 10000},
            },
            false, {},
            {       // Expected results.
            },
        },
        TestCase{
            "Real CIGAR, 1 window, aln ends in window",
            18, utils::Overlap{26888, 58586, 58586, 97, 31811, 75986, true}, "40=2I2=3X509=1X79=3D3=1D1X146=1I68=1X4=1X3=1I277=2D165=1I168=2D2=1X3=6D5=1D176=1I6=1I54=2X20=1I106=2X2=4X3=1X83=1D26=1X2=2X4=1I1X58=1X404=1D273=1X95=1I197=2I103=1I118=1I39=1D151=1X2=1X89=1D307=4I3=1D174=2I58=3D16=3D19=1I217=1I2=1I6=1I13=4I5=1X2=1X15=1I24=1I70=1I1=2X4=1D180=1D3=3D1=1X448=3I25=3I553=2X3=1D3=1X6=1I84=1X5=2D20=1D229=1D134=1X5=2D135=1X6=1I24=1D101=1X17=1I3=1X56=1I75=1D16=1D2=1X5=1I58=1I68=1X80=1I53=1D143=1X1=1D43=1X1=1X3=2X15=1I5=3D1=2X48=2X404=1X123=1X1=1X123=2D5=1X219=1I107=1D35=1D180=2X8=1D24=2I111=3I377=1I8=1I2=7I22=5D1205=1D2=1X138=8I59=14I313=4D82=4D414=1I10=2D5=1I4=1I5=1X230=1I687=1X343=1D15=1X3=2D80=4I309=1D54=1I36=3I822=1I435=1X1=1I3=1D76=1I71=2D48=1D191=4I129=1D51=1D246=3X94=1D3=1D146=2D15=1X1=1D88=1D10=1I9=1X657=1D419=1X182=1I3=1I21=1D48=1I1=2X210=1I1X88=1X7=2X1=1D1=2I635=1D3=3X81=2D122=1D96=3I3=1D72=1I236=4D53=1D2=1D50=7I205=2I4=1I77=1D42=1I103=1I127=1D683=1I75=4I1=1X4=1I1=1X9=1X4=1D2=1I1X2=1X88=1D1=1X3=1X1=1D68=4D84=4D70=2D36=2D106=1I3=1X2=1X129=2D6=3D122=4D27=2I26=2D21=2D141=2I81=2D64=2I46=1D130=1X3=1X117=1X5=1I1X1=1X21=1X496=1X80=1I12=1X35=2X4=1X153=1D4=1I5=1X23=1X1=1X97=1D96=1X6=1I3=1I73=2D65=1I344=1I101=1X116=1I81=3D72=4D142=2I88=2I13=1D1=1D5=1I27=2I20=6I52=2I46=2I59=2D202=1D59=2I68=2D38=2D75=2D137=1X2=2D47=6I92=2I17=2I22=2D77=2I79=3D14=2D37=2I23=1X6=1X133=6D133=2D105=4I63=1I27=2D120=4D198=1X5=1X26=1X43=1I5=1I217=1X263=2D10=1X23=1X145=1D159=4D38=1D18=1I4=1X3=1X2=3D2=1I1=1X15=1X96=2I43=1X91=2X3=1D1=1X44=1D14=1D5=2D3=1I196=1I2=1X14=1D3=1D27=2I124=2X139=2D48=2I176=2I13=8D20=4D44=4I26=3I1=1I3=1X9=2I91=2D12=2I67=1X10=4I46=1I104=2I60=2I185=2I70=2I76=1D3=1I7=1I22=2D12=2D68=2D44=1X2=2D20=2D58=2X1=1X37=1X1=1X123=2D170=6D36=2D107=6D46=2D59=1I23=2D19=4I257=1D110=2D89=1I7=1X4=4D11=1I51=2I220=1I162=1X10=1X97=1X30=1X50=1I2=1D372=1I2=1X78=1D21=2D29=3D27=1D1=1D1X121=1I11=2D9=1D195=1X1=1I117=2D2=1X6=1X166=1I1X3=1I99=2X68=4I94=",
            {
                Interval{28672, 32768},
            },
            false, {},
            {       // Expected results.
                {0, {18, 28672, 32768, 28672, 31811, 55468, 58586, 671, 20, 756, 0, 0.0, 3159}},
            },
        },
        TestCase{
            "Test using a custom CigarPoint (start cigar index, offset, target pos and query pos).",
            18, utils::Overlap{0, 1000, 1000, 6500, 7500, 10000, true}, "1000=",
            {
                Interval{0, 2000},
                Interval{7000, 10000},
            },
            true, CigarPoint{0, 500, 7000, 500},
            {       // Expected results.
                {1, {18, 7000, 10000, 7000, 7500, 500, 1000, 0, 500, 1, 0, 0.0, 500}},
            },
        },
    }));
    // clang-format on

    CATCH_INFO(TEST_GROUP << " Test name: " << test_case.test_name);

    const std::vector<CigarOp> cigar = parse_cigar_from_string(test_case.cigar);

#ifdef DEBUG_CORRECTION_WINDOW_TESTS
    std::cerr << "-------------------------------\n";
    std::cerr << "Test name: " << test_case.test_name << "\n";
#endif

    const std::vector<std::pair<int32_t, OverlapWindow>> result = split_alignment(
            test_case.overlap, cigar, test_case.window_intervals, test_case.aln_id, false,
            test_case.custom_initial_point, test_case.cigar_point.idx, test_case.cigar_point.offset,
            test_case.cigar_point.tpos, test_case.cigar_point.qpos);

#ifdef DEBUG_CORRECTION_WINDOW_TESTS
    std::cerr << "expected.size = " << std::size(test_case.expected) << "\n";
    for (size_t i = 0; i < std::size(test_case.expected); ++i) {
        std::cerr << "[expected i = " << i << "] win_id = " << test_case.expected[i].first
                  << ", window = {" << test_case.expected[i].second << "}\n";
    }
    std::cerr << "\n";
    std::cerr << "result.size = " << std::size(result) << "\n";
    for (size_t i = 0; i < std::size(result); ++i) {
        std::cerr << "[result i = " << i << "] win_id = " << result[i].first << ", window = {"
                  << result[i].second << "}\n";
    }
    std::cerr << "\n";
#endif

    CATCH_CHECK(test_case.expected == result);
}
}  // namespace dorado::correction::window_tests
