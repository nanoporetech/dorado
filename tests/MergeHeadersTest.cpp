#include "utils/MergeHeaders.h"

#include "TestUtils.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/sam.h>

#define TEST_GROUP "[merge_headers]"

using namespace dorado;
using dorado::utils::MergeHeaders;

CATCH_TEST_CASE("MergeHeadersTest: incompatible RG lines", TEST_GROUP) {
    std::string hdr1_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n";
    std::string hdr2_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY_ELSE\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n";
    SamHdrPtr header1(sam_hdr_parse(hdr1_txt.size(), hdr1_txt.c_str()));
    SamHdrPtr header2(sam_hdr_parse(hdr2_txt.size(), hdr2_txt.c_str()));

    MergeHeaders merger(true);
    auto res1 = merger.add_header(header1.get(), "header1");
    CATCH_CHECK(res1.empty());
    auto res2 = merger.add_header(header2.get(), "header2");
    CATCH_CHECK(res2 == std::string("Error merging header header2. RG lines are incompatible."));
}

CATCH_TEST_CASE("MergeHeadersTest: incompatible SQ lines", TEST_GROUP) {
    std::string hdr1_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n";
    std::string hdr2_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:2000\n";
    SamHdrPtr header1(sam_hdr_parse(hdr1_txt.size(), hdr1_txt.c_str()));
    SamHdrPtr header2(sam_hdr_parse(hdr2_txt.size(), hdr2_txt.c_str()));

    MergeHeaders merger(false);
    auto res1 = merger.add_header(header1.get(), "header1");
    CATCH_CHECK(res1.empty());
    auto res2 = merger.add_header(header2.get(), "header2");
    CATCH_CHECK(res2 == std::string("Error merging header header2. SQ lines are incompatible."));
}

CATCH_TEST_CASE("MergeHeadersTest: incompatible SQ lines, strip alignment", TEST_GROUP) {
    std::string hdr1_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n";
    std::string hdr2_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:2000\n";
    SamHdrPtr header1(sam_hdr_parse(hdr1_txt.size(), hdr1_txt.c_str()));
    SamHdrPtr header2(sam_hdr_parse(hdr2_txt.size(), hdr2_txt.c_str()));

    MergeHeaders merger(true);
    auto res1 = merger.add_header(header1.get(), "header1");
    CATCH_CHECK(res1.empty());
    auto res2 = merger.add_header(header2.get(), "header2");
    CATCH_CHECK(res2.empty());

    merger.finalize_merge();
    auto merged_hdr = merger.get_merged_header();
    std::string merged_hdr_txt = sam_hdr_str(merged_hdr);
    std::string expected_hdr_txt =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n";
    CATCH_CHECK(merged_hdr_txt == expected_hdr_txt);
}

CATCH_TEST_CASE("MergeHeadersTest: compatible header merge", TEST_GROUP) {
    std::string hdr1_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n";
    std::string hdr2_txt =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.27\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref2\tLN:3000\n";
    std::string hdr3_txt =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:demux\tPN:dorado\tVN:0.6.0\n"
            "@RG\tID:run2_model1\tDT:2022-12-20T10:35:17Z\tDS:runid=run2 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY_ELSE\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n"
            "@SQ\tSN:ref3\tLN:2000\n";
    SamHdrPtr header1(sam_hdr_parse(hdr1_txt.size(), hdr1_txt.c_str()));
    SamHdrPtr header2(sam_hdr_parse(hdr2_txt.size(), hdr2_txt.c_str()));
    SamHdrPtr header3(sam_hdr_parse(hdr3_txt.size(), hdr3_txt.c_str()));

    MergeHeaders merger(false);
    auto res1 = merger.add_header(header1.get(), "header1");
    CATCH_CHECK(res1.empty());
    auto res2 = merger.add_header(header2.get(), "header2");
    CATCH_CHECK(res2.empty());
    auto res3 = merger.add_header(header3.get(), "header3");
    CATCH_CHECK(res3.empty());

    merger.finalize_merge();
    auto merged_hdr = merger.get_merged_header();
    auto sq_mapping = merger.get_sq_mapping();

    std::string merged_hdr_txt = sam_hdr_str(merged_hdr);
    std::string expected_hdr_txt =
            "@HD\tVN:1.6\tSO:unknown\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26\n"
            "@PG\tID:aligner.1\tPN:minimap2\tVN:2.27\n"
            "@PG\tID:demux\tPN:dorado\tVN:0.6.0\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@RG\tID:run2_model1\tDT:2022-12-20T10:35:17Z\tDS:runid=run2 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY_ELSE\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n"
            "@SQ\tSN:ref2\tLN:3000\n"
            "@SQ\tSN:ref3\tLN:2000\n";
    CATCH_CHECK(merged_hdr_txt == expected_hdr_txt);
    CATCH_REQUIRE(sq_mapping.size() == 3);
    CATCH_REQUIRE(sq_mapping[0].size() == 1);
    CATCH_CHECK(sq_mapping[0][0] == 0);
    CATCH_REQUIRE(sq_mapping[1].size() == 1);
    CATCH_CHECK(sq_mapping[1][0] == 1);
    CATCH_REQUIRE(sq_mapping[2].size() == 2);
    CATCH_CHECK(sq_mapping[2][0] == 0);
    CATCH_CHECK(sq_mapping[2][1] == 2);
}
