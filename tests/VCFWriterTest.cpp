#include "TestUtils.h"
#include "hts_utils/hts_types.h"
#include "secondary/common/vcf_writer.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <htslib/hts.h>
#include <htslib/vcf.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace dorado::secondary::tests {

#define TEST_GROUP "[VCFWriter]"

namespace {

struct HtsFreeDeleter {
    void operator()(void* p) const { hts_free(p); }
};
using Int32BufferPtr = std::unique_ptr<int32_t, HtsFreeDeleter>;

std::vector<int32_t> decode_genotype(const int32_t* gt_data, const int gt_count) {
    std::vector<int32_t> decoded_gt;
    decoded_gt.reserve(gt_count);
    for (int i = 0; i < gt_count; ++i) {
        decoded_gt.emplace_back(bcf_gt_allele(gt_data[i]));
    }
    return decoded_gt;
}

void check_record(const bcf_hdr_t* header,
                  bcf1_t* record,
                  const Variant& expected,
                  const std::string_view expected_contig,
                  const std::vector<int32_t>& expected_gt,
                  const int32_t expected_gq) {
    CATCH_REQUIRE(bcf_unpack(record, BCF_UN_ALL) == 0);

    CATCH_CHECK(record->rid == expected.seq_id);
    CATCH_CHECK(bcf_hdr_id2name(header, record->rid) == expected_contig);
    CATCH_CHECK(record->pos == expected.pos);
    CATCH_REQUIRE(record->n_allele == (1 + std::size(expected.alts)));
    CATCH_CHECK(std::string(record->d.allele[0]) == expected.ref);
    for (int i = 0; i < std::ssize(expected.alts); ++i) {
        CATCH_CHECK(std::string(record->d.allele[i + 1]) == expected.alts[i]);
    }
    CATCH_CHECK(record->qual == Catch::Approx(expected.qual));

    for (std::string_view filt : expected.filters.as_strings()) {
        // Not a const ref because bcf_has_filter takes a non-const char pointer.
        std::string filter_name(filt);
        CATCH_CHECK(bcf_has_filter(header, record, filter_name.data()) == 1);
    }

    // Get genotype data, RAII-wrap it and check.
    int32_t* gt_raw = nullptr;
    int gt_capacity = 0;
    const int gt_count = bcf_get_genotypes(header, record, &gt_raw, &gt_capacity);
    Int32BufferPtr gt_values{gt_raw};
    CATCH_REQUIRE(gt_count == std::ssize(expected_gt));
    CATCH_CHECK(decode_genotype(gt_values.get(), gt_count) == expected_gt);

    // Get genotype quality data, RAII-wrap it and check.
    int32_t* gq_raw = nullptr;
    int gq_capacity = 0;
    const int gq_count = bcf_get_format_int32(header, record, "GQ", &gq_raw, &gq_capacity);
    Int32BufferPtr gq_values{gq_raw};
    CATCH_REQUIRE(gq_count == 1);
    CATCH_CHECK(gq_values.get()[0] == expected_gq);
}

}  // namespace

CATCH_TEST_CASE("VCFWriter writes valid VCF output and rejects invalid inputs", TEST_GROUP) {
    struct WriterTestData {
        std::filesystem::path vcf_path;
        Filters::FilterType filters;
        std::vector<std::pair<std::string, int64_t>> contigs;
        Variant first_variant;
        Variant second_variant;
        Variant third_variant;
    };

    const TempDir temp_dir = make_temp_dir("vcf_writer");

    const WriterTestData data{
            .vcf_path = temp_dir.m_path / "variants.vcf",
            .filters = Filters::Pass | Filters::LowQual | Filters::IncompatZyg,
            .contigs =
                    {
                            {"chr1", 100},
                            {"chr2", 250},
                    },
            .first_variant =
                    {
                            .seq_id = 0,
                            .pos = 9,
                            .ref = "A",
                            .alts = {"C"},
                            .filters = {Filters::Pass},
                            .info = {},
                            .qual = 60.0f,
                            .format = {{"GT", "0/1"}, {"GQ", "45"}},
                            .rstart = 0,
                            .rend = 0,
                    },
            .second_variant =
                    {
                            .seq_id = 1,
                            .pos = 41,
                            .ref = "T",
                            .alts = {"C", "G"},
                            .filters = {Filters::LowQual},
                            .info = {},
                            .qual = 17.5f,
                            .format = {{"GT", "1/2"},
                                       {"GQ", "31"},
                                       {"MIN_DP", "25"},
                                       {"AD", "1,14,10"},
                                       {"VAF", "0.56,0.4"},
                                       {"PL", "40,20,15,10,0,20"}},
                            .rstart = 0,
                            .rend = 0,
                    },
            // check correct handling of multiple variant filters
            .third_variant =
                    {
                            .seq_id = 1,
                            .pos = 51,
                            .ref = "T",
                            .alts = {"C"},
                            .filters = {Filters::LowQual | Filters::IncompatZyg},
                            .info = {},
                            .qual = 17.5f,
                            .format = {{"GT", "1"}, {"GQ", "31"}, {"MIN_DP", "25"}},
                            .rstart = 0,
                            .rend = 0,
                    },
    };

    CATCH_SECTION("round-trips header metadata and records through htslib") {
        // Write data to a file.
        {
            const std::string command_line = "dorado smallvar reads.bam reference.fa --device cpu";
            VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, command_line);
            writer.write_variant(data.first_variant);
            writer.write_variant(data.second_variant);
            writer.write_variant(data.third_variant);
        }

        // Check that the VCF text file contains the important lines.
        {
            CATCH_REQUIRE(std::filesystem::exists(data.vcf_path));
            CATCH_CHECK(std::filesystem::file_size(data.vcf_path) > 0);

            const std::string vcf_text = ReadFileIntoString(data.vcf_path);

            CATCH_CHECK_THAT(vcf_text,
                             Catch::Matchers::ContainsSubstring("##contig=<ID=chr1,length=100>"));
            CATCH_CHECK_THAT(vcf_text,
                             Catch::Matchers::ContainsSubstring("##contig=<ID=chr2,length=250>"));
            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                               "##FILTER=<ID=LowQual,Description=\"Variant quality "
                                               "is below threshold\">"));
            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                               "##FILTER=<ID=IncompatZyg,Description=\"Predicted "
                                               "zygosity incompatible with ploidy\">"));
            CATCH_CHECK_THAT(vcf_text,
                             Catch::Matchers::ContainsSubstring(
                                     "##FILTER=<ID=PASS,Description=\"All filters passed\">"));
            CATCH_CHECK_THAT(
                    vcf_text,
                    Catch::Matchers::ContainsSubstring(
                            "##cmdline=dorado smallvar reads.bam reference.fa --device cpu"));
            CATCH_CHECK_THAT(
                    vcf_text,
                    Catch::Matchers::ContainsSubstring(
                            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE"));
            CATCH_CHECK(vcf_text.find("##INFO=<ID=END") == std::string::npos);
            CATCH_CHECK(vcf_text.find("##ALT=<ID=*") == std::string::npos);
            CATCH_CHECK(vcf_text.find("##FORMAT=<ID=LEN") == std::string::npos);

            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring("chr1\t10\t."));
            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring("chr2\t42\t."));
            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring("LowQual;IncompatZyg"));
            CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                               "GT:GQ:MIN_DP:AD:VAF:PL:SRC\t1/"
                                               "2:31:25:1,14,10:0.56,0.4:40,20,15,10,0,20:."));
        }

        // Parse the VCF with Htslib to check that it is valid.
        {
            HtsFilePtr vcf_file{hts_open(data.vcf_path.string().c_str(), "r")};
            CATCH_REQUIRE(vcf_file);

            BcfHdrPtr header{bcf_hdr_read(vcf_file.get()), BcfHdrDestructor{}};

            CATCH_REQUIRE(header);
            CATCH_CHECK(bcf_hdr_nsamples(header.get()) == 1);
            CATCH_CHECK(std::string(header->samples[0]) == "SAMPLE");
            CATCH_CHECK(bcf_hdr_name2id(header.get(), "chr1") == 0);
            CATCH_CHECK(bcf_hdr_name2id(header.get(), "chr2") == 1);

            BcfRecordPtr first_record{bcf_init(), BcfRecordDestructor{}};
            BcfRecordPtr second_record{bcf_init(), BcfRecordDestructor{}};
            BcfRecordPtr third_record{bcf_init(), BcfRecordDestructor{}};

            CATCH_REQUIRE(first_record);
            CATCH_REQUIRE(second_record);
            CATCH_REQUIRE(third_record);

            CATCH_REQUIRE(bcf_read(vcf_file.get(), header.get(), first_record.get()) == 0);
            check_record(header.get(), first_record.get(), data.first_variant, "chr1", {0, 1}, 45);

            CATCH_REQUIRE(bcf_read(vcf_file.get(), header.get(), second_record.get()) == 0);
            check_record(header.get(), second_record.get(), data.second_variant, "chr2", {1, 2},
                         31);

            CATCH_REQUIRE(bcf_read(vcf_file.get(), header.get(), third_record.get()) == 0);
            check_record(header.get(), third_record.get(), data.third_variant, "chr2", {1}, 31);

            CATCH_CHECK(bcf_read(vcf_file.get(), header.get(), third_record.get()) < 0);
        }
    }

    // clang-format off
    CATCH_SECTION("writes gVCF header metadata when requested and tests that the writer sets the variant source FORMAT field to the missing value (\".\") when not provided") {
        {
            VCFWriter writer(data.vcf_path, data.filters, data.contigs, true, std::string_view{});
            writer.write_variant(data.first_variant);

            Variant no_call_variant{
                    .seq_id = 0,
                    .pos = 10,
                    .ref = "N",
                    .alts = {"<*>"},
                    .filters = {Filters::Missing},
                    .info = {{"END", "11"}},
                    .qual = -1.0f,
                    .format = {{"GT", "./."}, {"GQ", "."}, {"LEN", "1"}},   // NOTE: Intentionally not specifying the variant source here, to test that the writer adds it automatically.
                    .rstart = 10,
                    .rend = 11,
            };
            writer.write_variant(no_call_variant);
            writer.write_variant(data.second_variant);
        }

        const std::string vcf_text = ReadFileIntoString(data.vcf_path);

        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                           "##INFO=<ID=END,Number=1,Type=Integer,Description=\"End "
                                           "position of the reference block\">"));
        CATCH_CHECK_THAT(
                vcf_text,
                Catch::Matchers::ContainsSubstring(
                        "##FORMAT=<ID=LEN,Number=1,Type=Integer,Description=\"Length of <*> "
                        "reference block\">"));
        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring("chr1\t10\t.\tA\tC,<*>"));
        CATCH_CHECK_THAT(vcf_text,
                         Catch::Matchers::ContainsSubstring(
                                 "chr1\t11\t.\tN\t<*>\t.\t.\tEND=11\tGT:GQ:LEN:SRC\t./.:.:1:."));
        // Check that extra missing depth added for gVCF <*> record
        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                           "GT:GQ:MIN_DP:AD:VAF:PL:SRC\t1/2:31:25:1,14,10,.:0.56,0.4,.:40,20,15,10,0,20,.,.,.,.:."));
    }
    // clang-format on

    CATCH_SECTION(
            "preserves an explicit variant source FORMAT field instead of defaulting to the "
            "missing value") {
        {
            VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, std::string_view{});
            Variant variant = data.first_variant;
            variant.format = {
                    {"GT", "0/1"},
                    {"GQ", "45"},
                    {FORMAT_TAG_VARIANT_SOURCE, VARIANT_SOURCE_INFERENCE}};  // Set to "I".
            writer.write_variant(variant);
        }

        const std::string vcf_text = ReadFileIntoString(data.vcf_path);

        // The SRC FORMAT field is declared in the header.
        CATCH_CHECK_THAT(vcf_text,
                         Catch::Matchers::ContainsSubstring(
                                 "##FORMAT=<ID=SRC,Number=1,Type=String,Description=\"Source "
                                 "of the variant\">"));
        // The explicit source must be preserved and not overwritten with the missing-value default.
        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring("\tGT:GQ:SRC\t0/1:45:I"));
        CATCH_CHECK(vcf_text.find("0/1:45:.") == std::string::npos);
    }

    CATCH_SECTION("write_variant rejects filters missing from the header") {
        VCFWriter writer(data.vcf_path, Filters::LowQual, data.contigs, false, std::string_view{});

        CATCH_CHECK_THROWS_WITH(writer.write_variant(data.third_variant),
                                Catch::Matchers::ContainsSubstring("not found in header"));
    }

    CATCH_SECTION("write_variant rejects unexpected INFO fields") {
        VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, std::string_view{});
        Variant variant = data.first_variant;
        variant.info = {{"PIZZA", "1"}};

        CATCH_CHECK_THROWS_WITH(writer.write_variant(variant),
                                Catch::Matchers::ContainsSubstring("unexpected INFO field: PIZZA"));
    }

    CATCH_SECTION("write_variant rejects unexpected FORMAT fields") {
        VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, std::string_view{});
        Variant variant = data.first_variant;
        variant.format = {{"GT", "0/1"}, {"PIZZA", "1"}};

        CATCH_CHECK_THROWS_WITH(
                writer.write_variant(variant),
                Catch::Matchers::ContainsSubstring("unexpected FORMAT field: PIZZA"));
    }

    CATCH_SECTION("write_variant requires GT genotype information") {
        VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, std::string_view{});
        Variant variant = data.first_variant;
        variant.format = {{"GQ", "45"}};

        CATCH_CHECK_THROWS_WITH(
                writer.write_variant(variant),
                Catch::Matchers::ContainsSubstring("No genotype information found"));
    }

    CATCH_SECTION("writes_variant skips empty string") {
        {
            VCFWriter writer(data.vcf_path, data.filters, data.contigs, false, std::string_view{});

            Variant empty_list_variant{
                    .seq_id = 0,
                    .pos = 9,
                    .ref = "A",
                    .alts = {"C"},
                    .filters = {Filters::Pass},
                    .info = {},
                    .qual = 60.0f,
                    .format = {{"GT", "0/1"}, {"GQ", "45"}, {"AD", ""}},
                    .rstart = 0,
                    .rend = 0,
            };
            writer.write_variant(empty_list_variant);
        }

        const std::string vcf_text = ReadFileIntoString(data.vcf_path);

        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                           "##FORMAT=<ID=AD,Number=R,Type=Integer"));
        CATCH_CHECK_THAT(vcf_text, Catch::Matchers::ContainsSubstring(
                                           "chr1\t10\t.\tA\tC\t60\tPASS\t.\tGT:GQ:SRC\t0/1:45:."));
    }
}

}  // namespace dorado::secondary::tests
