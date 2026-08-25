#include "read_pipeline/base/ReadInitialiser.h"

#include "hts_utils/hts_types.h"
#include "utils/time_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/sam.h>

#include <string>

#define TEST_GROUP "[read_pipeline][ReadInitialiser]"

namespace {

dorado::SamHdrPtr make_header() {
    constexpr std::string_view header_text =
            "@HD\tVN:1.6\n"
            "@PG\tID:basecaller\tCL:dorado basecaller --min-qscore 10\n"
            "@RG\tID:read_group_barcode12\tPU:flowcell\tLB:sample\t"
            "DT:2024-01-02T03:04:05Z\t"
            "DS:runid=run-id experiment_id=experiment-id "
            "acquisition_start_time=2024-01-02T03:00:00Z model_stride=5\t"
            "tm:barcode\tSM:barcode12\tal:sample-12\tbk:SQK-RBK114-24\n";
    return dorado::SamHdrPtr{sam_hdr_parse(header_text.size(), header_text.data())};
}

dorado::HtsData make_data(std::string_view qname) {
    dorado::BamPtr record{bam_init1()};
    bam_set1(record.get(), qname.size(), qname.data(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0, 1, "*",
             "*", 0);
    return dorado::HtsData{std::move(record)};
}

void set_string_tag(dorado::HtsData& data, const char (&tag)[3], std::string_view value) {
    CATCH_REQUIRE(bam_aux_update_str(data.bam_ptr.get(), tag, static_cast<int>(value.size() + 1),
                                     value.data()) == 0);
}

std::string get_string_tag(const dorado::HtsData& data, const char (&tag)[3]) {
    const auto value = bam_aux_get(data.bam_ptr.get(), tag);
    CATCH_REQUIRE(value != nullptr);
    return bam_aux2Z(value);
}

}  // namespace

namespace dorado::test {

CATCH_TEST_CASE(TEST_GROUP " updates read attributes from a parsed header", TEST_GROUP) {
    auto header = make_header();
    CATCH_REQUIRE(header != nullptr);
    AlignmentCounts alignment_counts{{"read", {3, 1, 2}}};
    TrimFlags requested_trim;
    requested_trim.set_adapter();
    ReadInitialiser initialiser(header.get(), alignment_counts, requested_trim);

    auto data = make_data("read");
    set_string_tag(data, "RG", "read_group_barcode12");
    set_string_tag(data, "st", "2024-01-02T03:00:02Z");
    CATCH_REQUIRE(bam_aux_update_float(data.bam_ptr.get(), "qs", 9.5f) == 0);

    initialiser.update_read_attributes(data);
    initialiser.update_alignment_fields(data);

    CATCH_CHECK(data.read_attrs.model_stride == 5);
    CATCH_CHECK(data.read_attrs.flowcell_id == "flowcell");
    CATCH_CHECK(data.read_attrs.experiment_id == "experiment-id");
    CATCH_CHECK(data.read_attrs.sample_id == "sample");
    CATCH_CHECK(data.read_attrs.protocol_run_id == "run-id");
    CATCH_CHECK(data.read_attrs.protocol_start_time_ms ==
                utils::get_unix_time_ms_from_string_timestamp("2024-01-02T03:04:05Z"));
    CATCH_CHECK(data.read_attrs.start_time_ms == 2000);
    CATCH_CHECK_FALSE(data.read_attrs.is_status_pass);
    CATCH_CHECK(data.read_attrs.trim_flags.has_adapter());
    CATCH_CHECK(data.read_attrs.trim_flags.has_barcode());
    CATCH_CHECK(data.read_attrs.num_alignments == 3);
    CATCH_CHECK(data.read_attrs.num_secondary_alignments == 1);
    CATCH_CHECK(data.read_attrs.num_supplementary_alignments == 2);
}

CATCH_TEST_CASE(TEST_GROUP " updates record trim flags without a recognised read group",
                TEST_GROUP) {
    auto header = make_header();
    CATCH_REQUIRE(header != nullptr);
    TrimFlags requested_trim;
    requested_trim.set_primer();
    ReadInitialiser initialiser(header.get(), {}, requested_trim);

    auto data = make_data("read");
    set_string_tag(data, "RG", "unknown-read-group");
    set_string_tag(data, "tm", "adapter");

    initialiser.update_read_attributes(data);

    CATCH_CHECK(data.read_attrs.trim_flags.has_adapter());
    CATCH_CHECK(data.read_attrs.trim_flags.has_primer());
    CATCH_CHECK_FALSE(data.read_attrs.trim_flags.has_barcode());
}

CATCH_TEST_CASE(TEST_GROUP " updates barcoding fields from a parsed header", TEST_GROUP) {
    auto header = make_header();
    CATCH_REQUIRE(header != nullptr);
    ReadInitialiser initialiser(header.get(), {}, {});
    auto data = make_data("read");
    set_string_tag(data, "RG", "read_group_barcode12");

    initialiser.update_barcoding_fields(data);

    CATCH_REQUIRE(data.barcoding_result != nullptr);
    CATCH_CHECK(data.barcoding_result->barcode_name == "barcode12");
    CATCH_CHECK(data.barcoding_result->normalized_barcode_name == "barcode12");
    CATCH_CHECK(data.barcoding_result->alias == "sample-12");
    CATCH_CHECK(data.barcoding_result->kit == "SQK-RBK114-24");
}

CATCH_TEST_CASE(TEST_GROUP " undemultiplexes read groups using barcode aliases", TEST_GROUP) {
    auto header = make_header();
    CATCH_REQUIRE(header != nullptr);
    ReadInitialiser initialiser(header.get(), {}, {});
    auto data = make_data("read");
    set_string_tag(data, "RG", "read_group_sample-12");
    set_string_tag(data, "al", "sample-12");
    set_string_tag(data, "BC", "barcode12");

    initialiser.undemux_read_group(data);

    CATCH_CHECK(get_string_tag(data, "RG") == "read_group");
    CATCH_CHECK(bam_aux_get(data.bam_ptr.get(), "al") == nullptr);
    CATCH_CHECK(get_string_tag(data, "BC") == UNCLASSIFIED_STR);
}

CATCH_TEST_CASE(TEST_GROUP " undemultiplexes read groups using barcode tags", TEST_GROUP) {
    auto header = make_header();
    CATCH_REQUIRE(header != nullptr);
    ReadInitialiser initialiser(header.get(), {}, {});
    auto data = make_data("read");
    set_string_tag(data, "RG", "read_group_barcode12");
    set_string_tag(data, "BC", "barcode12");

    initialiser.undemux_read_group(data);

    CATCH_CHECK(get_string_tag(data, "RG") == "read_group");
    CATCH_CHECK(get_string_tag(data, "BC") == UNCLASSIFIED_STR);
}

}  // namespace dorado::test
