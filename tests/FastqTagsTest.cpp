#include "TestUtils.h"
#include "hts_utils/fastq_tags.h"

#include <catch2/catch_test_macros.hpp>

#include <ostream>
#include <tuple>

namespace dorado::utils {

inline bool operator==(const dorado::utils::ReadGroupData& lhs,
                       const dorado::utils::ReadGroupData& rhs) {
    return std::tie(lhs.found, lhs.id, lhs.data.basecalling_model, lhs.data.device_id,
                    lhs.data.exp_start_time, lhs.data.experiment_id, lhs.data.flowcell_id,
                    lhs.data.modbase_models, lhs.data.position_id, lhs.data.run_id,
                    lhs.data.sample_id) == std::tie(rhs.found, rhs.id, rhs.data.basecalling_model,
                                                    rhs.data.device_id, rhs.data.exp_start_time,
                                                    rhs.data.experiment_id, rhs.data.flowcell_id,
                                                    rhs.data.modbase_models, rhs.data.position_id,
                                                    rhs.data.run_id, rhs.data.sample_id);
}

inline std::ostream& operator<<(std::ostream& os, const dorado::utils::ReadGroupData& data) {
    os << "found = " << data.found << ", id = '" << data.id
       << "', data = {run_id = " << data.data.run_id
       << ", basecalling_model = " << data.data.basecalling_model
       << ", modbase_models = " << data.data.modbase_models
       << ", flowcell_id = " << data.data.flowcell_id << ", device_id = " << data.data.device_id
       << ", exp_start_time = " << data.data.exp_start_time
       << ", sample_id = " << data.data.sample_id << ", position_id = " << data.data.position_id
       << ", experiment_id = " << data.data.experiment_id << "}\n";
    return os;
}

}  // namespace dorado::utils

CATCH_TEST_CASE("Empty input", "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected;

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Normal tags, tab separated, run_id UUID style",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0	"
            "ch:i:63	st:Z:2022-10-18T10:38:07.247+00:00	PU:Z:PAM93185	LB:Z:PCR_zymo	"
            "SM:Z:barcode03	al:Z:alias_for_bc03	"
            "pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5	DS:Z:gpu:Tesla V100-PCIE-16GB	"
            "qs:f:30.0	mx:i:2	rn:i:432	ts:i:1048	pt:i:120	"
            "pa:B:i,12,30,45,232,242"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            .data =
                    {
                            .run_id = "4524e8b9-b90e-4ffb-a13a-380266513b64",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                            .flowcell_id = "PAM93185",
                            .exp_start_time = "2022-10-18T10:38:07.247+00:00",
                            .sample_id = "PCR_zymo",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Normal tags, tab separated, protocol_id UUID style",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "@f5f11551-1f50-4f75-a5a9-ade67e104a60	NM:i:337	ms:i:9265	"
            "AS:i:9200	nn:i:0	tp:A:P	cm:i:497	s1:i:3376	s2:i:2609	"
            "de:f:0.0485	SA:Z:contig_49,34690,+,1813M25D5590S,60,136;	rl:i:1862	"
            "qs:f:12.4662	du:f:20.0252	ns:i:100126	ts:i:460	mx:i:3	"
            "ch:i:958	st:Z:2023-11-02T21:22:44.417+00:00	rn:i:108109	"
            "fn:Z:PAS25963_pass_a432b9f0_bc8993f4_2186.pod5	sm:f:725.842	sd:f:125.658	"
            "sv:Z:pa	dx:i:0	NM:i:446	ms:i:12338	AS:i:12238	nn:i:0	"
            "de:f:0.0469106	tp:A:P	cm:i:552	s1:i:3847	s2:i:2249	"
            "rl:i:2530	"
            "RG:Z:bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            .data =
                    {
                            .run_id = "bc8993f4557dd53bf0cbda5fd68453fea5e94485",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                            .exp_start_time = "2023-11-02T21:22:44.417+00:00",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Suffix after the basecaller version, UUID is run_id style",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-"
            "barcode01	"
            "ch:i:63	st:Z:2022-10-18T10:38:07.247+00:00	PU:Z:PAM93185	LB:Z:PCR_zymo	"
            "SM:Z:barcode03	al:Z:alias_for_bc03	"
            "pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5	DS:Z:gpu:Tesla V100-PCIE-16GB	"
            "qs:f:30.0	mx:i:2	rn:i:432	ts:i:1048	pt:i:120	"
            "pa:B:i,12,30,45,232,242"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-"
                  "barcode01",
            .data =
                    {
                            .run_id = "4524e8b9-b90e-4ffb-a13a-380266513b64",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                            .flowcell_id = "PAM93185",
                            .exp_start_time = "2022-10-18T10:38:07.247+00:00",
                            .sample_id = "PCR_zymo",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Suffix after the basecaller version, UUID is protocol_id style",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "@f5f11551-1f50-4f75-a5a9-ade67e104a60	NM:i:337	ms:i:9265	"
            "AS:i:9200	nn:i:0	tp:A:P	cm:i:497	s1:i:3376	s2:i:2609	"
            "de:f:0.0485	SA:Z:contig_49,34690,+,1813M25D5590S,60,136;	rl:i:1862	"
            "qs:f:12.4662	du:f:20.0252	ns:i:100126	ts:i:460	mx:i:3	"
            "ch:i:958	st:Z:2023-11-02T21:22:44.417+00:00	rn:i:108109	"
            "fn:Z:PAS25963_pass_a432b9f0_bc8993f4_2186.pod5	sm:f:725.842	sd:f:125.658	"
            "sv:Z:pa	dx:i:0	NM:i:446	ms:i:12338	AS:i:12238	nn:i:0	"
            "de:f:0.0469106	tp:A:P	cm:i:552	s1:i:3847	s2:i:2249	"
            "rl:i:2530	"
            "RG:Z:bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-"
            "barcode01"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "bc8993f4557dd53bf0cbda5fd68453fea5e94485_dna_r10.4.1_e8.2_400bps_hac@v5.0.0-"
                  "barcode01",
            .data =
                    {
                            .run_id = "bc8993f4557dd53bf0cbda5fd68453fea5e94485",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                            .exp_start_time = "2023-11-02T21:22:44.417+00:00",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Normal tags, only RG", "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            .data =
                    {
                            .run_id = "4524e8b9-b90e-4ffb-a13a-380266513b64",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Normal tags, mix of HTS-style and non-HTS tokens is allowed",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "nonhts	"
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0	"
            "another_non	htsch:i:63	st:Z:2022-10-18T10:38:07.247+00:00	"
            "PU:Z:PAM93185	LB:Z:PCR_zymo	SM:Z:barcode03	al:Z:alias_for_bc03	"
            "pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5	DS:Z:gpu:Tesla V100-PCIE-16GB	"
            "qs:f:30.0	mx:i:2	rn:i:432	ts:i:1048	pt:i:120	"
            "pa:B:i,12,30,45,232,242"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
            .data =
                    {
                            .run_id = "4524e8b9-b90e-4ffb-a13a-380266513b64",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                            .flowcell_id = "PAM93185",
                            .exp_start_time = "2022-10-18T10:38:07.247+00:00",
                            .sample_id = "PCR_zymo",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Space separated tags don't work", "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0 ch:i:63 "
            "st:Z:2022-10-18T10:38:07.247+00:00 PU:Z:PAM93185 LB:Z:PCR_zymo SM:Z:barcode03 "
            "al:Z:alias_for_bc03 pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5 DS:Z:gpu:Tesla "
            "V100-PCIE-16GB qs:f:30.0 mx:i:2 rn:i:432 ts:i:1048 pt:i:120 pa:B:i,12,30,45,232,242"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    // Since there are no tabs, everything is considered as the first tag that was at the very front of the string (the RG).
    // The basecaller model and the run_id will still be parsed here because they will match the pattern.
    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0 ch:i:63 "
                  "st:Z:2022-10-18T10:38:07.247+00:00 PU:Z:PAM93185 LB:Z:PCR_zymo SM:Z:barcode03 "
                  "al:Z:alias_for_bc03 pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5 DS:Z:gpu:Tesla "
                  "V100-PCIE-16GB qs:f:30.0 mx:i:2 rn:i:432 ts:i:1048 pt:i:120 "
                  "pa:B:i,12,30,45,232,242",
            .data =
                    {
                            .run_id = "4524e8b9-b90e-4ffb-a13a-380266513b64",
                            .basecalling_model = "dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
                    },
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("No matching HTS-style tags", "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{"device_id=cuda:2"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected;

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("RG does not match the pattern, basecaller model cannot be parsed",
                "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{"RG:Z:4524e8b9-dna_r10.4.1_e8.2_400bps_hac@v5.0.0"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = false,
            .id = "4524e8b9-dna_r10.4.1_e8.2_400bps_hac@v5.0.0",
    };

    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("Cannot parse an 'unknown' model.", "fastq_tags parse_rg_from_hts_tags") {
    using namespace dorado::utils;

    constexpr std::string_view IN_STR{
            "RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_unknown_barcode01	"
            "ch:i:63	st:Z:2022-10-18T10:38:07.247+00:00	PU:Z:PAM93185	LB:Z:PCR_zymo	"
            "SM:Z:barcode03	al:Z:alias_for_bc03	"
            "pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5	DS:Z:gpu:Tesla V100-PCIE-16GB	"
            "qs:f:30.0	mx:i:2	rn:i:432	ts:i:1048	pt:i:120	"
            "pa:B:i,12,30,45,232,242"};

    const ReadGroupData result = parse_rg_from_hts_tags(IN_STR);

    const ReadGroupData expected{
            .found = true,
            .id = "4524e8b9-b90e-4ffb-a13a-380266513b64_unknown_barcode01",
            .data =
                    {
                            .flowcell_id = "PAM93185",
                            .exp_start_time = "2022-10-18T10:38:07.247+00:00",
                            .sample_id = "PCR_zymo",
                    },
    };

    CATCH_CHECK(result == expected);
}
