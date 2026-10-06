#include "TestUtils.h"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "secondary/common/bam_info.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#define TEST_GROUP "[SecondaryBamInfo]"

namespace dorado::secondary::tests {
namespace {

struct BamRecord {
    std::string_view qname{};
    std::vector<uint8_t> dwell_tag{};
    std::string_view read_group{};
};

// write a synthetic bam file
std::filesystem::path write_bam(const std::filesystem::path& path,
                                const std::string_view rg_headers,
                                const std::vector<BamRecord>& records) {
    const std::string header_text =
            "@HD\tVN:1.6\n"
            "@SQ\tSN:chr1\tLN:100\n"
            "@SQ\tSN:chr2\tLN:200\n"
            "@PG\tID:aligner\tPP:samtools\tPN:dorado\n"
            "@PG\tID:basecaller\tPN:dorado\n" +
            std::string(rg_headers);
    SamHdrPtr header{sam_hdr_parse(std::size(header_text), header_text.c_str())};
    CATCH_REQUIRE(header != nullptr);
    utils::HtsFile output(path.string(), utils::HtsFile::OutputMode::BAM, 1, false);
    output.set_header(header.get());

    for (const BamRecord& record : records) {
        BamPtr ret{bam_init1()};
        CATCH_REQUIRE(bam_set1(ret.get(), std::ssize(record.qname), record.qname.data(), BAM_FUNMAP,
                               -1, -1, 0, 0, nullptr, -1, -1, 0, 0, nullptr, nullptr, 0) >= 0);
        if (!std::empty(record.dwell_tag)) {
            std::vector<uint8_t> aux_data;
            aux_data.emplace_back('c');

            const uint32_t n = static_cast<uint32_t>(std::size(record.dwell_tag));
            const uint8_t* n_bytes = reinterpret_cast<const uint8_t*>(&n);
            aux_data.insert(std::end(aux_data), n_bytes, n_bytes + sizeof(n));

            for (const int8_t val : record.dwell_tag) {
                aux_data.emplace_back(static_cast<uint8_t>(val));
            }
            const int32_t aux_data_len = static_cast<int32_t>(std::size(aux_data));
            CATCH_REQUIRE(bam_aux_append(ret.get(), "mv", 'B', aux_data_len, std::data(aux_data)) ==
                          0);
        }
        if (!std::empty(record.read_group)) {
            CATCH_REQUIRE(bam_aux_update_str(ret.get(), "RG", std::ssize(record.read_group),
                                             record.read_group.data()) == 0);
        }
        CATCH_REQUIRE(output.write(ret.get()) >= 0);
    }
    output.finalise([](size_t) {});
    CATCH_REQUIRE(sam_index_build(path.string().c_str(), 0) == 0);
    return path;
}

}  // namespace

CATCH_TEST_CASE("Analyze BAM headers and dwell tags", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("secondary_bam_info");
    const std::string_view rg_headers =
            "@RG\tID:rg-a\tDS:runid=a basecall_model=model-a\tLB:sample-a\n"
            "@RG\tID:rg-b\tDS:basecall_model=model-a\tLB:sample-a\n"
            "@RG\tID:rg-c\tDS:basecall_model=model-b\tLB:sample-b\n"
            "@RG\tID:rg-d\tDS:basecall_model=model-c\n"
            "@RG\tID:no-model\tDS:runid=d\tLB:ignored\n";

    const std::unordered_set<std::string> expected_read_groups{"rg-a", "rg-b", "rg-c", "rg-d"};
    const std::unordered_set<std::string> expected_models{"model-a", "model-b", "model-c"};
    const std::unordered_set<std::string> expected_sample_names{"sample-a", "sample-b"};
    const std::vector<std::pair<std::string, int64_t>> expected_refs{{"chr1", 100}, {"chr2", 200}};

    CATCH_SECTION("First read has no dwells") {
        const std::vector<BamRecord> records{
                {"read_01", {}, "rg-a"},
                {"read_02", {5, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0}, "rg-b"},
        };
        const auto bam_path = write_bam(temp_dir.m_path / "input_base.bam", rg_headers, records);
        const BamInfo info = analyze_bam(bam_path, "");
        CATCH_CHECK(info.uses_dorado_aligner);
        CATCH_CHECK_FALSE(info.has_dwells);
        CATCH_CHECK(info.read_groups == expected_read_groups);
        CATCH_CHECK(info.basecaller_models == expected_models);
        CATCH_CHECK(info.sample_names == expected_sample_names);
        CATCH_CHECK(info.ref_seqs == expected_refs);
    }
    CATCH_SECTION("First read has dwells") {
        const std::vector<BamRecord> records{
                {"read_02", {5, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0}, "rg-b"},
                {"read_01", {}, "rg-a"},
        };
        const auto bam_path = write_bam(temp_dir.m_path / "input_base.bam", rg_headers, records);
        const BamInfo info = analyze_bam(bam_path, "");
        CATCH_CHECK(info.uses_dorado_aligner);
        CATCH_CHECK(info.has_dwells);
        CATCH_CHECK(info.read_groups == expected_read_groups);
        CATCH_CHECK(info.basecaller_models == expected_models);
        CATCH_CHECK(info.sample_names == expected_sample_names);
        CATCH_CHECK(info.ref_seqs == expected_refs);
    }
}

CATCH_TEST_CASE("check_read_groups", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("secondary_bam_info");
    CATCH_SECTION("bam contains one RG") {
        const std::string_view rg_headers_one =
                "@RG\tID:rg-a\tDS:runid=a basecall_model=model-a\tLB:sample-a\n";
        const auto bam_path = write_bam(temp_dir.m_path / "input_one_rg.bam", rg_headers_one, {});
        const BamInfo bam_info = analyze_bam(bam_path, "");
        CATCH_CHECK_NOTHROW(check_read_groups(bam_info, ""));
        CATCH_CHECK_NOTHROW(check_read_groups(bam_info, "rg-a"));
        CATCH_CHECK_THROWS_WITH(
                check_read_groups(bam_info, "rg-b"),
                Catch::Matchers::ContainsSubstring("Requested RG is not in the input BAM"));
    }
    CATCH_SECTION("bam contains no valid RG") {
        const std::string_view rg_headers_invalid = "@RG\tID:no-model\tDS:runid=d\tLB:ignored\n";
        const auto bam_path =
                write_bam(temp_dir.m_path / "input_invalid_rg.bam", rg_headers_invalid, {});
        const BamInfo bam_info = analyze_bam(bam_path, "");
        CATCH_CHECK_NOTHROW(check_read_groups(bam_info, ""));
        CATCH_CHECK_THROWS_WITH(check_read_groups(bam_info, "rg-b"),
                                Catch::Matchers::ContainsSubstring("No @RG headers found"));
    }
    CATCH_SECTION("bam contains multiple RGs") {
        const std::string_view rg_headers_multiple =
                "@RG\tID:rg-b\tDS:basecall_model=model-a\tLB:sample-a\n"
                "@RG\tID:rg-c\tDS:basecall_model=model-b\tLB:sample-b\n"
                "@RG\tID:rg-d\tDS:basecall_model=model-c\n";
        const auto bam_path =
                write_bam(temp_dir.m_path / "input_multi_rg.bam", rg_headers_multiple, {});
        const BamInfo bam_info = analyze_bam(bam_path, "");
        CATCH_CHECK_THROWS_WITH(check_read_groups(bam_info, ""),
                                Catch::Matchers::ContainsSubstring("more than one read group"));
        CATCH_CHECK_NOTHROW(check_read_groups(bam_info, "rg-b"));
        CATCH_CHECK_THROWS_WITH(
                check_read_groups(bam_info, "rg-a"),
                Catch::Matchers::ContainsSubstring("Requested RG is not in the input BAM"));
    }
}

CATCH_TEST_CASE("unique_sample_name_or_throw", TEST_GROUP) {
    const TempDir temp_dir = make_temp_dir("secondary_bam_info");
    const std::string_view rg_headers =
            "@RG\tID:rg-a\tDS:runid=a basecall_model=model-a\tLB:sample-a\n"
            "@RG\tID:rg-b\tDS:basecall_model=model-a\tLB:sample-a\n"
            "@RG\tID:rg-c\tDS:basecall_model=model-b\tLB:sample-b\n"
            "@RG\tID:rg-d\tDS:basecall_model=model-c\n"
            "@RG\tID:no-model\tDS:runid=d\tLB:ignored\n";
    const auto bam_path = write_bam(temp_dir.m_path / "input_test_sample_name.bam", rg_headers, {});
    CATCH_SECTION("RGs contains multiple sample_names") {
        const BamInfo bam_info = analyze_bam(bam_path, "");
        CATCH_CHECK_THROWS_WITH(
                unique_sample_name_or_throw(bam_info),
                Catch::Matchers::ContainsSubstring("contain multiple sample names"));
    }
    CATCH_SECTION("RGs contains no sample_names") {
        const BamInfo bam_info = analyze_bam(bam_path, "rg-d");
        CATCH_CHECK_THROWS_WITH(unique_sample_name_or_throw(bam_info),
                                Catch::Matchers::ContainsSubstring("contain no sample names"));
    }
    CATCH_SECTION("RGs contain one sample_name") {
        const BamInfo bam_info = analyze_bam(bam_path, "rg-a");
        CATCH_CHECK(unique_sample_name_or_throw(bam_info) == "sample-a");
    }
}

}  // namespace dorado::secondary::tests
