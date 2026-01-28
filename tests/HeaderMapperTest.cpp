#include "hts_utils/HeaderMapper.h"

#include "TestUtils.h"
#include "hts_utils/hts_types.h"
#include "read_pipeline/base/HtsReader.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <unordered_map>

#define TEST_GROUP "[header_mapper]"

namespace fs = std::filesystem;

namespace {
void check_read_attrs(const dorado::HtsData::ReadAttributes &result,
                      const dorado::HtsData::ReadAttributes &expected) {
    CATCH_CHECK(result.sequencing_kit == expected.sequencing_kit);
    CATCH_CHECK(result.experiment_id == expected.experiment_id);
    CATCH_CHECK(result.sample_id == expected.sample_id);
    CATCH_CHECK(result.position_id == expected.position_id);
    CATCH_CHECK(result.flowcell_id == expected.flowcell_id);
    CATCH_CHECK(result.protocol_run_id == expected.protocol_run_id);
    CATCH_CHECK(result.acquisition_id == expected.acquisition_id);
    CATCH_CHECK(result.protocol_start_time_ms == expected.protocol_start_time_ms);
    CATCH_CHECK(result.subread_id == expected.subread_id);
    CATCH_CHECK(result.is_status_pass == expected.is_status_pass);
}

fs::path write_bam_without_rg(const fs::path &output_dir, const std::string &filename) {
    auto bam_path = output_dir / filename;
    dorado::HtsFilePtr file(hts_open(bam_path.string().c_str(), "wb"));
    CATCH_REQUIRE(file != nullptr);

    dorado::SamHdrPtr header(sam_hdr_init());
    CATCH_REQUIRE(header != nullptr);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "HD", "VN", "1.6", "SO", "unknown", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "SQ", "SN", "ref", "LN", "10", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_write(file.get(), header.get()) == 0);

    dorado::BamPtr record(bam_init1());
    const std::string qname{"read1"};
    bam_set1(record.get(), qname.size(), qname.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0, 1, "*",
             "*", 0);
    CATCH_REQUIRE(sam_write1(file.get(), header.get(), record.get()) >= 0);

    return bam_path;
}
}  // namespace

namespace dorado::utils::test {

CATCH_TEST_CASE(TEST_GROUP " parse multiple inputs", TEST_GROUP) {
    auto sam = fs::path(get_data_dir("aligner_test")) / "basecall.sam";
    auto bam = fs::path(get_data_dir("hts_file")) / "test_data.bam";

    // @RG	ID:a16f403b6a3655419511bf356ce3b40b65abfae4_dna_r9.4.1_e8_hac@v3.3	PU:PAK21298	PM:PAPAP48
    // DT:2022-04-27T16:47:57.305+00:00	PL:ONT	DS:basecall_model=dna_r9.4.1_e8_hac@v3.3 modbase_models=dna_r9.4.1_e8_hac@v3.3_5mCG@v0.1
    // runid=a16f403b6a3655419511bf356ce3b40b65abfae4	LB:no_sample	SM:no_sample
    const HtsData::ReadAttributes expected_basecall_attr{
            "",
            "",
            "no_sample",
            "0",
            "PAK21298",
            "a16f403b6a3655419511bf356ce3b40b65abfae4",
            "0000000000000000000000000000000000000000",
            "",
            "",
            1651078077305,
            0,
            true,
    };

    // @RG ID:0a73e955b30dc4b0182e1abb710bca268b16d689_dna_r10.4.1_e8.2_400bps_sup@v4.2.0 PU:PAO83395 PM:PC24B318
    // DT:2023-04-29T16:06:40.107+00:00 PL:ONT DS:basecall_model=dna_r10.4.1_e8.2_400bps_sup@v4.2.0
    // runid=0a73e955b30dc4b0182e1abb710bca268b16d689 LB:PrePCR SM:PrePCR
    const HtsData::ReadAttributes expected_test_data_attr{
            "",
            "",
            "PrePCR",
            "0",
            "PAO83395",
            "0a73e955b30dc4b0182e1abb710bca268b16d689",
            "0000000000000000000000000000000000000000",
            "",
            "",
            1682784400107,
            0,
            true,
    };

    const std::unordered_map<std::string, HtsData::ReadAttributes> expected{
            {"a16f403b6a3655419511bf356ce3b40b65abfae4_dna_r9.4.1_e8_hac@v3.3",
             expected_basecall_attr},
            {"0a73e955b30dc4b0182e1abb710bca268b16d689_dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
             expected_test_data_attr}};

    utils::HeaderMapper mapper({sam, bam}, false);
    const auto &result_attrs_map = *mapper.get_read_attributes_map();
    CATCH_CHECK(result_attrs_map.size() == expected.size());

    for (const auto &[expected_id, expected_attrs] : expected) {
        CATCH_SECTION("Checking: '" + expected_id + "'") {
            CATCH_REQUIRE(result_attrs_map.contains(expected_id));

            const auto &result_attrs = result_attrs_map.at(expected_id);
            check_read_attrs(result_attrs, expected_attrs);

            const auto &merged_header = mapper.get_merged_header(result_attrs);
            const auto sam_header = merged_header.get_merged_header();
            CATCH_REQUIRE(sam_header != nullptr);
        }
    }
}

CATCH_TEST_CASE(TEST_GROUP " fallback for BAM without RG lines", TEST_GROUP) {
    auto temp_dir = dorado::tests::make_temp_dir("header_mapper_no_rg");
    auto bam_path = write_bam_without_rg(temp_dir.m_path, "no_rg.bam");

    utils::HeaderMapper mapper({bam_path}, false);
    HtsReader reader(bam_path.string(), std::nullopt);
    CATCH_REQUIRE(reader.read());

    const auto &attrs = mapper.get_read_attributes(reader.record.get());
    const auto &merged_header = mapper.get_merged_header(attrs);

    CATCH_CHECK_NOTHROW(merged_header.get_sq_mapping(bam_path.string()));
    const auto &mapping = merged_header.get_sq_mapping(bam_path.string());
    CATCH_CHECK(mapping.size() == 1);
}

CATCH_TEST_CASE(TEST_GROUP " fallback merges multiple BAMs without RG lines", TEST_GROUP) {
    // Create two BAMS without RG lines
    auto temp_dir = dorado::tests::make_temp_dir("header_mapper_multi_no_rg");
    auto first_bam = write_bam_without_rg(temp_dir.m_path, "no_rg_one.bam");
    auto second_bam = write_bam_without_rg(temp_dir.m_path, "no_rg_two.bam");

    // Map headers for both files
    utils::HeaderMapper mapper({first_bam, second_bam}, false);

    // Load attrs and headers of each
    HtsReader first_reader(first_bam.string(), std::nullopt);
    CATCH_REQUIRE(first_reader.read());
    const auto &first_attrs = mapper.get_read_attributes(first_reader.record.get());
    const auto &first_header = mapper.get_merged_header(first_attrs);

    HtsReader second_reader(second_bam.string(), std::nullopt);
    CATCH_REQUIRE(second_reader.read());
    const auto &second_attrs = mapper.get_read_attributes(second_reader.record.get());
    const auto &second_header = mapper.get_merged_header(second_attrs);

    // Expect headers to be the same instance
    CATCH_CHECK(&first_header == &second_header);

    // The header should include both files
    CATCH_CHECK_NOTHROW(first_header.get_sq_mapping(first_bam.string()));
    CATCH_CHECK_NOTHROW(first_header.get_sq_mapping(second_bam.string()));

    // There should be more extra records (to unknown files)
    CATCH_CHECK(first_header.get_sq_mapping().size() == 2);

    // There should be no extra records in the maps
    CATCH_CHECK(first_header.get_sq_mapping(first_bam.string()).size() == 1);
    CATCH_CHECK(first_header.get_sq_mapping(second_bam.string()).size() == 1);
}

};  // namespace dorado::utils::test
