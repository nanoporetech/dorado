#include "hts_utils/HeaderMapper.h"

#include "TestUtils.h"
#include "hts_utils/hts_types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

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

};  // namespace dorado::utils::test