#include "hts_utils/HeaderMapper.h"

#include "TestUtils.h"
#include "hts_utils/KString.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_types.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/base/messages.h"
#include "read_pipeline/base/terminate_options.h"

#include <catch2/catch_test_macros.hpp>
#include <htslib/sam.h>

#include <array>
#include <filesystem>
#include <set>
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
    CATCH_CHECK(result.barcode_id == expected.barcode_id);
    CATCH_CHECK(result.barcode_alias == expected.barcode_alias);
    CATCH_CHECK(result.trim_flags.m_flags ==
                expected.trim_flags.m_flags);  // Check the inner value for catch2 logging purposes
}

fs::path write_file_with_barcode_tags(const fs::path &output_dir,
                                      const std::string &filename,
                                      const std::string &alias,
                                      const std::string &format) {
    auto bam_path = output_dir / filename;
    dorado::HtsFilePtr file(hts_open(bam_path.string().c_str(), format.c_str()));
    CATCH_REQUIRE(file != nullptr);

    constexpr std::array fastq_aux_tags{"RG", "PU", "DT", "SM", "al", "LB"};
    for (const auto &tag : fastq_aux_tags) {
        hts_set_opt(file.get(), FASTQ_OPT_AUX, tag);
    }

    dorado::SamHdrPtr header(sam_hdr_init());
    const std::string read_group = format + "_with_barcodes" + alias;
    const std::string barcode_id = "barcode01";
    const std::string flowcell_id = "PAK21298";
    CATCH_REQUIRE(header != nullptr);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "HD", "VN", "1.6", "SO", "unknown", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_add_line(
                          header.get(), "RG", "ID", read_group.c_str(), "PU", flowcell_id.c_str(),
                          "PM", "PAPAP48", "LB", format.c_str(), "SM", barcode_id.c_str(), "al",
                          alias.empty() ? barcode_id.c_str() : alias.c_str(), nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_write(file.get(), header.get()) == 0);

    dorado::BamPtr record(bam_init1());
    const std::string qname{"read1"};
    bam_set1(record.get(), qname.size(), qname.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0, 1, "*",
             "*", 0);
    bam_aux_update_str(record.get(), "RG", static_cast<int>(read_group.length() + 1),
                       read_group.c_str());
    bam_aux_update_str(record.get(), "PU", static_cast<int>(flowcell_id.length() + 1),
                       flowcell_id.c_str());
    bam_aux_update_str(record.get(), "LB", static_cast<int>(format.length() + 1), format.c_str());
    bam_aux_update_str(record.get(), "SM", static_cast<int>(barcode_id.length() + 1),
                       barcode_id.c_str());
    const std::string &alias_tag = alias.empty() ? barcode_id : alias;
    bam_aux_update_str(record.get(), "al", static_cast<int>(alias_tag.length() + 1),
                       alias_tag.c_str());
    CATCH_REQUIRE(sam_write1(file.get(), header.get(), record.get()) >= 0);

    return bam_path;
}

fs::path write_bam_with_barcode_tags(const fs::path &output_dir,
                                     const std::string &filename,
                                     const std::string &alias) {
    return write_file_with_barcode_tags(output_dir, filename, alias, "wb");
}

fs::path write_fastq_with_barcode_tags(const fs::path &output_dir,
                                       const std::string &filename,
                                       const std::string &alias) {
    return write_file_with_barcode_tags(output_dir, filename, alias, "wf");
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

fs::path write_bam_with_read_group(const fs::path &output_dir,
                                   const std::string &filename,
                                   const std::string &read_group_id,
                                   const std::string &platform_model,
                                   const std::string &qname) {
    auto bam_path = output_dir / filename;
    dorado::HtsFilePtr file(hts_open(bam_path.string().c_str(), "wb"));
    CATCH_REQUIRE(file != nullptr);

    dorado::SamHdrPtr header(sam_hdr_init());
    CATCH_REQUIRE(header != nullptr);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "HD", "VN", "1.6", "SO", "unknown", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "SQ", "SN", "ref", "LN", "10", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_add_line(header.get(), "RG", "ID", read_group_id.c_str(), "PU",
                                   "PAK21298", "PM", platform_model.c_str(), "LB", "sample", "DS",
                                   "runid=run-1 experiment_id=exp-1", nullptr) == 0);
    CATCH_REQUIRE(sam_hdr_write(file.get(), header.get()) == 0);

    dorado::BamPtr record(bam_init1());
    bam_set1(record.get(), qname.size(), qname.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0, 1, "*",
             "*", 0);
    bam_aux_update_str(record.get(), "RG", static_cast<int>(read_group_id.length() + 1),
                       read_group_id.c_str());
    CATCH_REQUIRE(sam_write1(file.get(), header.get(), record.get()) >= 0);

    return bam_path;
}

class BamCollectorNode : public dorado::MessageSink {
public:
    BamCollectorNode() : MessageSink(10, 1) {}
    ~BamCollectorNode() override {
        stop_input_processing(dorado::utils::AsyncQueueTerminateFast::Yes);
    }

    std::string get_name() const override { return "BamCollector"; }
    void terminate(const dorado::TerminateOptions &terminate_options) override {
        stop_input_processing(terminate_options.fast);
    }
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "bamcollector_node");
    }

    const std::vector<dorado::BamMessage> &messages() const { return m_messages; }

private:
    BamCollectorNode(const BamCollectorNode &) = delete;
    BamCollectorNode &operator=(const BamCollectorNode &) = delete;
    BamCollectorNode(BamCollectorNode &&) = delete;
    BamCollectorNode &operator=(BamCollectorNode &&) = delete;

private:
    void input_thread_fn() {
        dorado::Message message;
        while (get_input_message(message)) {
            if (message.holds<dorado::BamMessage>()) {
                m_messages.push_back(message.take<dorado::BamMessage>());
            }
        }
    }

    std::vector<dorado::BamMessage> m_messages;
};
}  // namespace

namespace dorado::utils::test {

CATCH_TEST_CASE(TEST_GROUP " parse multiple inputs", TEST_GROUP) {
    auto sam = fs::path(get_data_dir("aligner_test")) / "basecall.sam";
    auto bam = fs::path(get_data_dir("hts_file")) / "test_data.bam";
    auto temp_dir = dorado::tests::make_temp_dir("header_mapper_inputs");
    auto fastq_barcodes = write_fastq_with_barcode_tags(temp_dir.m_path, "with_barcodes.fastq", "");
    auto bam_barcodes = write_bam_with_barcode_tags(temp_dir.m_path, "with_barcodes.bam", "");
    auto bam_alias = write_bam_with_barcode_tags(temp_dir.m_path, "with_barcodes_and_alias.bam",
                                                 "_and_alias");

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
            "no_sample",
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
            "PrePCR",
            "",
            1682784400107,
            0,
            true,
    };

    const HtsData::ReadAttributes expected_barcodes_attr{
            "",
            "",
            "wb",
            "0",
            "PAK21298",
            "00000000-0000-0000-0000-000000000000",
            "0000000000000000000000000000000000000000",
            "barcode01",
            "",
            0,
            0,
            true,
    };

    const HtsData::ReadAttributes expected_alias_attr{
            "",
            "",
            "wb",
            "0",
            "PAK21298",
            "00000000-0000-0000-0000-000000000000",
            "0000000000000000000000000000000000000000",
            "barcode01",
            "_and_alias",
            0,
            0,
            true,
    };

    const HtsData::ReadAttributes expected_fastq_attr{
            "",
            "",
            "wf",
            "0",
            "PAK21298",
            "00000000-0000-0000-0000-000000000000",
            "0000000000000000000000000000000000000000",
            "barcode01",
            "",
            0,
            0,
            true,
    };

    const std::unordered_map<std::string, HtsData::ReadAttributes> expected{
            {"a16f403b6a3655419511bf356ce3b40b65abfae4_dna_r9.4.1_e8_hac@v3.3",
             expected_basecall_attr},
            {"0a73e955b30dc4b0182e1abb710bca268b16d689_dna_r10.4.1_e8.2_400bps_sup@v4.2.0",
             expected_test_data_attr},
            {"wb_with_barcodes", expected_barcodes_attr},
            {"wb_with_barcodes_and_alias", expected_alias_attr},
            {"wf_with_barcodes", expected_fastq_attr},
    };

    utils::HeaderMapper mapper({sam, bam, bam_barcodes, bam_alias, fastq_barcodes}, std::nullopt,
                               nullptr, false);
    const auto &result_attrs_map = mapper.get_read_attributes_map();
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

    utils::HeaderMapper mapper({bam_path}, std::nullopt, nullptr, false);
    TestHtsReader reader(bam_path.string());
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
    utils::HeaderMapper mapper({first_bam, second_bam}, std::nullopt, nullptr, false);

    // Load attrs and headers of each
    TestHtsReader first_reader(first_bam.string());
    CATCH_REQUIRE(first_reader.read());
    const auto &first_attrs = mapper.get_read_attributes(first_reader.record.get());
    const auto &first_header = mapper.get_merged_header(first_attrs);

    TestHtsReader second_reader(second_bam.string());
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

CATCH_TEST_CASE(TEST_GROUP " maps read groups to attributes", TEST_GROUP) {
    std::unordered_map<std::string, ReadGroup> read_groups;

    ReadGroup rg_one{
            .run_id = "run-1",
            .flowcell_id = "flow-1",
            .exp_start_time = "2022-04-27T16:47:57.305+00:00",
            .sample_id = "sample-1",
            .position_id = "pos-1",
            .experiment_id = "exp-1",
    };
    read_groups.emplace("rg-one", rg_one);

    ReadGroup rg_two{
            .flowcell_id = "flow-2",
            .sample_id = "sample-2",
            .trim_flags = TrimFlags::from_string("adapter,primer"),
    };
    read_groups.emplace("rg-two", rg_two);

    utils::HeaderMapper mapper(read_groups, std::nullopt, nullptr);
    const auto &result_attrs_map = mapper.get_read_attributes_map();
    CATCH_REQUIRE(result_attrs_map.size() == 2);

    const HtsData::ReadAttributes expected_one{
            .sequencing_kit = "",
            .experiment_id = "exp-1",
            .sample_id = "sample-1",
            .position_id = "pos-1",
            .flowcell_id = "flow-1",
            .protocol_run_id = "run-1",
            .acquisition_id = "0000000000000000000000000000000000000000",
            .barcode_id = "",
            .barcode_alias = "",
            .protocol_start_time_ms = 1651078077305,
            .subread_id = 0,
            .is_status_pass = true,
            .trim_flags = 0,
    };
    const HtsData::ReadAttributes expected_two{
            .sequencing_kit = "",
            .experiment_id = "",
            .sample_id = "sample-2",
            .position_id = "0",
            .flowcell_id = "flow-2",
            .protocol_run_id = "00000000-0000-0000-0000-000000000000",
            .acquisition_id = "0000000000000000000000000000000000000000",
            .barcode_id = "",
            .barcode_alias = "",
            .protocol_start_time_ms = 0,
            .subread_id = 0,
            .is_status_pass = true,
            .trim_flags = 3,
    };

    CATCH_REQUIRE(result_attrs_map.contains("rg-one"));
    check_read_attrs(result_attrs_map.at("rg-one"), expected_one);
    CATCH_REQUIRE(result_attrs_map.contains("rg-two"));
    check_read_attrs(result_attrs_map.at("rg-two"), expected_two);
}

CATCH_TEST_CASE(TEST_GROUP " remaps conflicting read group ids by filename", TEST_GROUP) {
    auto temp_dir = dorado::tests::make_temp_dir("header_mapper_rg_conflicts");
    const auto original_read_group_id = std::string{"shared_rg"};
    auto first_bam = write_bam_with_read_group(temp_dir.m_path, "first.bam", original_read_group_id,
                                               "device-one", "read1");
    auto second_bam = write_bam_with_read_group(temp_dir.m_path, "second.bam",
                                                original_read_group_id, "device-two", "read2");

    utils::HeaderMapper mapper({first_bam, second_bam}, std::nullopt, nullptr, false);

    CATCH_CHECK(mapper.get_output_read_group_id(first_bam.string(), original_read_group_id) ==
                original_read_group_id);
    const auto remapped_read_group_id =
            mapper.get_output_read_group_id(second_bam.string(), original_read_group_id);
    CATCH_CHECK(remapped_read_group_id == "shared_rg_1");

    const auto &read_attrs_map = mapper.get_read_attributes_map();
    CATCH_REQUIRE(read_attrs_map.contains(original_read_group_id));
    CATCH_REQUIRE(read_attrs_map.contains("shared_rg_1"));
    check_read_attrs(read_attrs_map.at(original_read_group_id), read_attrs_map.at("shared_rg_1"));

    const auto &merged_header = mapper.get_merged_header(read_attrs_map.at(original_read_group_id));
    std::set<std::string> read_group_ids;
    dorado::SamHdrPtr sam_header(sam_hdr_dup(merged_header.get_merged_header()));
    const int num_rg_lines = sam_hdr_count_lines(sam_header.get(), "RG");
    CATCH_REQUIRE(num_rg_lines == 2);
    for (int idx = 0; idx < num_rg_lines; ++idx) {
        read_group_ids.insert(sam_hdr_line_name(sam_header.get(), "RG", idx));
    }
    CATCH_CHECK(read_group_ids == std::set<std::string>{original_read_group_id, "shared_rg_1"});
}

CATCH_TEST_CASE(TEST_GROUP " rewrites record read group ids using remap lookup", TEST_GROUP) {
    auto temp_dir = dorado::tests::make_temp_dir("header_mapper_rg_record_remap");
    const auto original_read_group_id = std::string{"shared_rg"};
    auto first_bam = write_bam_with_read_group(temp_dir.m_path, "first.bam", original_read_group_id,
                                               "device-one", "read1");
    auto second_bam = write_bam_with_read_group(temp_dir.m_path, "second.bam",
                                                original_read_group_id, "device-two", "read2");

    utils::HeaderMapper mapper({first_bam, second_bam}, std::nullopt, nullptr, false);
    const auto remapped_read_group_id =
            mapper.get_output_read_group_id(second_bam.string(), original_read_group_id);
    CATCH_REQUIRE(remapped_read_group_id == "shared_rg_1");

    TestHtsReader reader(second_bam.string());
    dorado::PipelineDescriptor pipeline_desc;
    const auto collector_handle = pipeline_desc.add_node<BamCollectorNode>({});
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
    CATCH_REQUIRE(pipeline != nullptr);

    CATCH_REQUIRE(reader.read(*pipeline, 1, false, &mapper, false) == 1);
    pipeline->terminate({.fast = dorado::utils::AsyncQueueTerminateFast::No});

    const auto &collector = pipeline->get_node_ref<BamCollectorNode>(collector_handle);
    CATCH_REQUIRE(collector.messages().size() == 1);

    const auto &message = collector.messages().front();
    CATCH_CHECK(dorado::utils::get_read_group_tag(message.data->bam_ptr.get()) == "shared_rg_1");
    CATCH_CHECK(message.data->read_attrs == mapper.get_read_attributes_map().at("shared_rg_1"));
}

CATCH_TEST_CASE(TEST_GROUP " barcode kit adds barcoded read groups headers", TEST_GROUP) {
    std::unordered_map<std::string, ReadGroup> read_groups;

    ReadGroup rg_one{};
    rg_one.sample_id = "sample-1";
    read_groups.emplace("rg-one", rg_one);

    ReadGroup rg_two{};
    rg_two.sample_id = "sample-2";
    read_groups.emplace("rg-two", rg_two);

    utils::HeaderMapper mapper(read_groups, "SQK-RBK114-24", nullptr);
    const auto merged_headers_map = mapper.get_merged_headers_map();
    CATCH_REQUIRE(merged_headers_map != nullptr);
    CATCH_CHECK(merged_headers_map->size() ==
                75);  // (2 read groups + fallback) * (24 barcodes + unclassified) = 3 * 25

    auto merged_header = mapper.get_shared_merged_header(true);
    int num_read_groups = sam_hdr_count_lines(merged_header.get(), "RG");
    CATCH_CHECK(num_read_groups == 50);  // no fallback
}

CATCH_TEST_CASE(TEST_GROUP " read group trim flags", TEST_GROUP) {
    std::unordered_map<std::string, ReadGroup> read_groups;

    ReadGroup rg_one{
            .sample_id = "sample-1",
            .trim_flags = 3,
    };
    read_groups.emplace("rg-one", rg_one);

    ReadGroup rg_two{
            .sample_id = "sample-2",
            .trim_flags = 0,
    };
    read_groups.emplace("rg-two", rg_two);

    utils::HeaderMapper mapper(read_groups, std::nullopt, nullptr);
    const auto merged_headers_map = mapper.get_merged_headers_map();
    CATCH_REQUIRE(merged_headers_map != nullptr);
    CATCH_CHECK(merged_headers_map->size() == 3);  // (2 read groups + fallback)

    auto merged_header = mapper.get_shared_merged_header(true);
    int num_read_groups = sam_hdr_count_lines(merged_header.get(), "RG");
    CATCH_CHECK(num_read_groups == 2);  // no fallback

    KString tag_wrapper(100000);
    auto &tag_value = tag_wrapper.get();
    for (int idx = 0; idx < num_read_groups; ++idx) {
        CATCH_REQUIRE(sam_hdr_find_tag_pos(merged_header.get(), "RG", idx, "tm", &tag_value) >= 0);
        std::string tm_str{tag_value.s, tag_value.l};
        std::string rg_id = sam_hdr_line_name(merged_header.get(), "RG", idx);
        CATCH_CHECK(tm_str == to_string(read_groups[rg_id].trim_flags));
    }
}

};  // namespace dorado::utils::test
