#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "alignment/Minimap2Aligner.h"
#include "alignment/alignment_info.h"
#include "alignment/minimap2_args.h"
#include "alignment/minimap2_wrappers.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
#include "utils/concurrency/multi_queue_thread_pool.h"
#include "utils/sequence_utils.h"
#include "utils/string_utils.h"
#include "utils/types.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#define TEST_GROUP "[bam_utils][aligner]"

using Catch::Matchers::Equals;
namespace fs = std::filesystem;

namespace {

std::unordered_map<std::string, std::string> get_tags_from_sam_line_fields(
        const std::vector<std::string>& fields) {
    // returns tags TAG:TYPE:VALUE as kvps of TAG:TYPE to VALUE
    std::unordered_map<std::string, std::string> result{};
    // sam line is 11 fields plus tags
    CHECK(fields.size() >= 11);
    for (std::size_t field_index{11}; field_index < fields.size(); ++field_index) {
        const auto& field = fields[field_index];
        auto tokens = dorado::utils::split(field, ':');
        CHECK(tokens.size() == 3);
        result[tokens[0] + ":" + tokens[1]] = tokens[2];
    }
    return result;
}

class AlignerNodeTestFixture {
private:
    std::vector<dorado::Message> m_output_messages{};

protected:
    std::unique_ptr<dorado::Pipeline> pipeline;
    dorado::NodeHandle aligner_node_handle;

    template <class... Args>
    void create_pipeline(Args&&... args) {
        dorado::PipelineDescriptor pipeline_desc;
        auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, m_output_messages);
        aligner_node_handle = pipeline_desc.add_node<dorado::AlignerNode>({sink}, args...);
        pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);
    }

    std::vector<dorado::BamPtr> RunPipelineWithBamMessages(
            dorado::HtsReader& reader,
            const std::string& reference_file,
            const std::string& bed_file,
            const dorado::alignment::Minimap2Options& options,
            int threads) {
        auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
        auto bed_file_access = std::make_shared<dorado::alignment::BedFileAccess>();
        if (!bed_file.empty()) {
            bed_file_access->load_bedfile(bed_file);
        }
        create_pipeline(index_file_access, bed_file_access, reference_file, bed_file, options,
                        threads);

        auto client_info = std::make_shared<dorado::DefaultClientInfo>();
        auto alignment_info = std::make_shared<dorado::alignment::AlignmentInfo>();
        alignment_info->minimap_options = options;
        alignment_info->reference_file = reference_file;
        client_info->contexts().register_context<const dorado::alignment::AlignmentInfo>(
                alignment_info);
        reader.set_client_info(client_info);

        reader.read(*pipeline, 100);
        pipeline->terminate({});
        auto bam_messages = ConvertMessages<dorado::BamMessage>(std::move(m_output_messages));
        std::vector<dorado::BamPtr> result{};
        result.reserve(bam_messages.size());
        for (auto& bam_message : bam_messages) {
            result.push_back(std::move(bam_message.bam_ptr));
        }
        return result;
    }

    template <class MessageType, class MessageTypePtr = std::unique_ptr<MessageType>>
    MessageTypePtr RunPipelineForRead(
            const std::shared_ptr<dorado::alignment::AlignmentInfo>& loaded_align_info,
            const std::shared_ptr<dorado::alignment::AlignmentInfo>& client_align_info,
            std::string read_id,
            std::string sequence) {
        auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
        auto bed_file_access = std::make_shared<dorado::alignment::BedFileAccess>();
        CHECK(index_file_access->load_index(loaded_align_info->reference_file,
                                            loaded_align_info->minimap_options,
                                            2) == dorado::alignment::IndexLoadResult::success);
        auto thread_pool = std::make_shared<dorado::utils::concurrency::MultiQueueThreadPool>(2);
        create_pipeline(index_file_access, bed_file_access, thread_pool,
                        dorado::utils::concurrency::TaskPriority::normal);

        auto client_info = std::make_shared<dorado::DefaultClientInfo>();
        client_info->contexts().register_context<const dorado::alignment::AlignmentInfo>(
                client_align_info);

        auto read = std::make_unique<MessageType>();
        read->read_common.client_info = std::move(client_info);
        read->read_common.read_id = std::move(read_id);
        read->read_common.seq = std::move(sequence);

        pipeline->push_message(std::move(read));
        pipeline->terminate({});

        CHECK((m_output_messages.size() == 1 &&
               std::holds_alternative<MessageTypePtr>(m_output_messages[0])));

        return std::get<MessageTypePtr>(std::move(m_output_messages[0]));
    }

    std::string get_sam_line_from_bam(dorado::BamPtr bam_record) {
        dorado::SamHdrPtr header(sam_hdr_init());
        const auto& aligner_ref =
                dynamic_cast<dorado::AlignerNode&>(pipeline->get_node_ref(aligner_node_handle));
        dorado::utils::add_sq_hdr(header.get(), aligner_ref.get_sequence_records_for_header());

        dorado::KString line_wrapper(1000000);
        kstring_t sam_line_buffer = line_wrapper.get();
        CHECK(sam_format1(header.get(), bam_record.get(), &sam_line_buffer) >= 0);

        return std::string(ks_str(&sam_line_buffer), ks_len(&sam_line_buffer));
    }
};

}  // namespace

TEST_CASE_METHOD(AlignerNodeTestFixture, "AlignerTest: Check standard alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "target.fq";

    auto options = dorado::alignment::create_dflt_options();
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();
    bam1_t* in_rec = reader.record.get();

    // Check input/output reads are matching.
    std::string orig_read = dorado::utils::extract_sequence(in_rec);
    std::string aligned_read = dorado::utils::extract_sequence(rec);

    // Check quals are matching.
    std::vector<uint8_t> orig_qual = dorado::utils::extract_quality(in_rec);
    std::vector<uint8_t> aligned_qual = dorado::utils::extract_quality(rec);
    CHECK(orig_qual == aligned_qual);

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"NMi", "msi", "ASi", "nni", "def", "tpA", "cmi", "s1i", "rli"};
    for (const auto& tag : tags) {
        CHECK_THAT(aux, Contains(tag));
    }
}

TEST_CASE_METHOD(AlignerNodeTestFixture, "AlignerTest: Check alignment with bed file", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "target.fq";
    auto bed = aligner_test_dir / "target.bed";

    auto options = dorado::alignment::create_dflt_options();
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), bed.string(), options, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();
    bam1_t* in_rec = reader.record.get();

    // Check input/output reads are matching.
    std::string orig_read = dorado::utils::extract_sequence(in_rec);
    std::string aligned_read = dorado::utils::extract_sequence(rec);

    // Check quals are matching.
    std::vector<uint8_t> orig_qual = dorado::utils::extract_quality(in_rec);
    std::vector<uint8_t> aligned_qual = dorado::utils::extract_quality(rec);
    CHECK(orig_qual == aligned_qual);

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"NMi", "msi", "ASi", "nni", "def", "tpA", "cmi", "s1i", "rli", "bhi"};
    for (const auto& tag : tags) {
        CHECK_THAT(aux, Contains(tag));
    }
    auto bh_tag_ptr = bam_aux_get(rec, "bh");
    auto bh_tag_type = bam_aux_type(bh_tag_ptr);
    CHECK(bh_tag_type == 'i');
    auto bh_tag_value = bam_aux2i(bh_tag_ptr);
    CHECK(bh_tag_value == 3);
}

TEST_CASE_METHOD(AlignerNodeTestFixture, "AlignerTest: Check supplementary alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_aln_target.fa";
    auto query = aligner_test_dir / "supplementary_aln_query.fa";

    auto options = dorado::alignment::create_dflt_options();
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 2);

    // Check first alignment is primary.
    {
        bam1_t* rec = bam_records[0].get();

        // Check aux tags.
        uint32_t l_aux = bam_get_l_aux(rec);
        std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
        CHECK_THAT(aux, Contains("tpAP"));
        CHECK(rec->core.l_qseq > 0);  // Primary alignment should have SEQ.
    }

    // Check second alignment is secondary.
    {
        bam1_t* rec = bam_records[1].get();

        // Check aux tags.
        uint32_t l_aux = bam_get_l_aux(rec);
        std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
        CHECK_THAT(aux, Contains("tpAS"));
        CHECK(rec->core.l_qseq == 0);  // Secondary alignment doesn't need SEQ.
    }
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerTest: Check reverse complement alignment",
                 TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "rev_target.fq";

    auto options = dorado::alignment::create_dflt_options();
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();
    bam1_t* in_rec = reader.record.get();

    // Check flag.
    CHECK(rec->core.flag & 0x10);

    // Check read reverse complementing.
    std::string orig_read = dorado::utils::extract_sequence(in_rec);
    std::string aligned_read = dorado::utils::extract_sequence(rec);
    CHECK(orig_read == dorado::utils::reverse_complement(aligned_read));

    // Check qual reversal.
    std::vector<uint8_t> orig_qual = dorado::utils::extract_quality(in_rec);
    std::vector<uint8_t> aligned_qual = dorado::utils::extract_quality(rec);
    std::reverse(aligned_qual.begin(), aligned_qual.end());
    CHECK(orig_qual == aligned_qual);
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerTest: Check dorado tags are retained",
                 TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "basecall_target.fa";
    auto query = aligner_test_dir / "basecall.sam";

    auto options = dorado::alignment::create_dflt_options();
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"RGZ", "MMZ", "MLB"};
    for (const auto& tag : tags) {
        CHECK_THAT(aux, Contains(tag));
    }
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerTest: Check modbase tags are removed for secondary alignments",
                 TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_basecall_target.fa";
    auto query = aligner_test_dir / "basecall.sam";

    auto options = dorado::alignment::create_dflt_options();
    options.index_options->get().k = 15;
    options.index_options->get().w = 15;
    bool soft_clipping = GENERATE(true, false);
    if (soft_clipping) {
        options.mapping_options->get().flag |= MM_F_SOFTCLIP;
    }

    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 3);

    bam1_t* primary_rec = bam_records[0].get();
    bam1_t* secondary_rec = bam_records[1].get();
    bam1_t* supplementary_rec = bam_records[2].get();

    // Check aux tags.
    if (soft_clipping) {
        CHECK(bam_aux_get(primary_rec, "MM") != nullptr);
        CHECK(bam_aux_get(primary_rec, "ML") != nullptr);
        CHECK(bam_aux_get(primary_rec, "MN") != nullptr);
        CHECK(bam_aux_get(secondary_rec, "MM") != nullptr);
        CHECK(bam_aux_get(secondary_rec, "ML") != nullptr);
        CHECK(bam_aux_get(secondary_rec, "MN") != nullptr);
        CHECK(bam_aux_get(supplementary_rec, "MM") != nullptr);
        CHECK(bam_aux_get(supplementary_rec, "ML") != nullptr);
        CHECK(bam_aux_get(supplementary_rec, "MN") != nullptr);
    } else {
        CHECK(bam_aux_get(primary_rec, "MM") != nullptr);
        CHECK(bam_aux_get(primary_rec, "ML") != nullptr);
        CHECK(bam_aux_get(primary_rec, "MN") != nullptr);
        CHECK(bam_aux_get(secondary_rec, "MM") == nullptr);
        CHECK(bam_aux_get(secondary_rec, "ML") == nullptr);
        CHECK(bam_aux_get(secondary_rec, "MN") == nullptr);
        CHECK(bam_aux_get(supplementary_rec, "MM") == nullptr);
        CHECK(bam_aux_get(supplementary_rec, "ML") == nullptr);
        CHECK(bam_aux_get(supplementary_rec, "MN") == nullptr);
    }
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerTest: Verify impact of updated aligner args",
                 TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "query.fa";

    // Run alignment with one set of k/w.
    {
        auto options = dorado::alignment::mm2::parse_options("-k 28 -w 28");
        dorado::HtsReader reader(query.string(), std::nullopt);
        auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 2);
        CHECK(bam_records.size() == 2);  // Generates 2 alignments.
    }

    // Run alignment with another set of k/w.
    {
        auto options = dorado::alignment::mm2::parse_options("-k 5 -w 5");
        dorado::HtsReader reader(query.string(), std::nullopt);
        auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 2);
        CHECK(bam_records.size() == 1);  // Generates 1 alignment.
    }
}

TEST_CASE("AlignerTest: Check AlignerNode crashes if multi index encountered", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "long_target.fa";

    auto options = dorado::alignment::mm2::parse_options("-k 5 -w 5 -I 1K");
    auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
    auto bed_file_access = std::make_shared<dorado::alignment::BedFileAccess>();
    CHECK_THROWS(
            dorado::AlignerNode(index_file_access, bed_file_access, ref.string(), "", options, 1));
}

SCENARIO_METHOD(AlignerNodeTestFixture, "AlignerNode push SimplexRead", TEST_GROUP) {
    GIVEN("AlgnerNode constructed with populated index file collection") {
        const std::string READ_ID{"aligner_node_test"};
        fs::path aligner_test_dir{get_aligner_data_dir()};
        auto ref = aligner_test_dir / "target.fq";

        auto align_info = std::make_shared<dorado::alignment::AlignmentInfo>();
        align_info->minimap_options = dorado::alignment::create_dflt_options();
        align_info->reference_file = ref.string();

        const std::string TEST_SEQUENCE{"ACGTACGTACGTACGT"};
        const std::string TEST_QUALITY{""};  // deliberately empty
        const std::string POSTFIX = "\t" + TEST_SEQUENCE + "\t" + TEST_QUALITY;

        AND_GIVEN("client with no alignment requirements") {
            const auto EMPTY_ALIGN_INFO = std::make_shared<dorado::alignment::AlignmentInfo>();
            WHEN("push simplex read to pipeline") {
                auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(
                        align_info, EMPTY_ALIGN_INFO, READ_ID, TEST_SEQUENCE);

                THEN("Output simplex read has empty alignments") {
                    REQUIRE(simplex_read->read_common.alignment_results.empty());
                }
            }

            WHEN("push duplex read to pipeline") {
                auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(
                        align_info, EMPTY_ALIGN_INFO, READ_ID, TEST_SEQUENCE);
                THEN("Output duplex read has empty alignments") {
                    REQUIRE(duplex_read->read_common.alignment_results.empty());
                }
            }
        }

        AND_GIVEN("client requiring alignment") {
            WHEN("push simplex read with no alignment matches to pipeline") {
                auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(align_info, align_info,
                                                                            READ_ID, TEST_SEQUENCE);

                THEN("Output simplex read has alignments populated") {
                    REQUIRE_FALSE(simplex_read->read_common.alignment_results.empty());
                }

                THEN("Output simplex read has alignment_string containing unmapped sam line") {
                    std::string expected{READ_ID + dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                    expected.insert(expected.size() - 1, POSTFIX);
                    std::string sam_string;
                    for (const auto& result : simplex_read->read_common.alignment_results) {
                        sam_string += result.sam_string + "\n";
                    }
                    REQUIRE(sam_string == expected);
                }
            }

            WHEN("push duplex read with no alignment matches to pipeline") {
                auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(align_info, align_info,
                                                                          READ_ID, TEST_SEQUENCE);

                THEN("Output duplex read has alignment_string populated") {
                    REQUIRE_FALSE(duplex_read->read_common.alignment_results.empty());
                }

                THEN("Output duplex read has alignment_string containing unmapped sam line") {
                    std::string expected{READ_ID + dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                    expected.insert(expected.size() - 1, POSTFIX);
                    std::string sam_string;
                    for (const auto& result : duplex_read->read_common.alignment_results) {
                        sam_string += result.sam_string + "\n";
                    }
                    REQUIRE(sam_string == expected);
                }
            }

            AND_GIVEN("read with alignment matches") {
                dorado::HtsReader reader(ref.string(), std::nullopt);
                reader.read();
                auto sequence = dorado::utils::extract_sequence(reader.record.get());

                WHEN("pushed as simplex read to pipeline") {
                    auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(
                            align_info, align_info, READ_ID, sequence);

                    THEN("Output sam line has read_id as QNAME") {
                        const std::string expected{READ_ID +
                                                   dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                        REQUIRE(simplex_read->read_common.alignment_results[0].sam_string.substr(
                                        0, READ_ID.size()) == READ_ID);
                    }

                    THEN("Output sam line contains sequence string") {
                        REQUIRE_FALSE(
                                simplex_read->read_common.alignment_results[0].sam_string.find(
                                        sequence) == std::string::npos);
                    }
                }

                WHEN("pushed as duplex read to pipeline") {
                    auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(
                            align_info, align_info, READ_ID, sequence);

                    THEN("Output sam line has read_id as QNAME") {
                        const std::string expected{READ_ID +
                                                   dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                        REQUIRE(duplex_read->read_common.alignment_results[0].sam_string.substr(
                                        0, READ_ID.size()) == READ_ID);
                    }

                    THEN("Output sam line contains sequence string") {
                        REQUIRE_FALSE(duplex_read->read_common.alignment_results[0].sam_string.find(
                                              sequence) == std::string::npos);
                    }
                }
            }
        }
    }
}

std::pair<std::string, std::string> get_read_id_and_sequence_from_fasta(
        const std::string& fasta_file) {
    std::ifstream query_input_stream(fasta_file);

    std::string line;
    std::getline(query_input_stream, line);
    CHECK(!line.empty());
    CHECK(dorado::utils::starts_with(line, ">"));
    line = line.substr(1);
    auto read_id = line.substr(0, line.find_first_of(' '));

    std::string sequence;
    while (std::getline(query_input_stream, sequence)) {
        if (!dorado::utils::starts_with(sequence, ">")) {
            break;
        }
    }
    CHECK(!sequence.empty());

    return {read_id, sequence};
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerNode compare BamPtr and ReadCommon processing",
                 TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = (aligner_test_dir / "target.fq").string();
    auto query = (aligner_test_dir / "query.fa").string();

    auto options = dorado::alignment::mm2::parse_options("-k 5 -w 5");

    // Get the sam line from BAM pipeline
    dorado::HtsReader bam_reader(query, std::nullopt);
    bam_reader.set_add_filename_tag(false);
    auto bam_records = RunPipelineWithBamMessages(bam_reader, ref, "", options, 2);
    CHECK(bam_records.size() == 1);
    auto sam_line_from_bam_ptr = get_sam_line_from_bam(std::move(bam_records[0]));

    // Get the sam line from ReadCommon pipeline
    auto [read_id, sequence] = get_read_id_and_sequence_from_fasta(query);
    auto align_info = std::make_shared<dorado::alignment::AlignmentInfo>();
    align_info->minimap_options = options;
    align_info->reference_file = ref;
    auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(
            align_info, align_info, std::move(read_id), std::move(sequence));
    auto sam_line_from_read_common =
            std::move(simplex_read->read_common.alignment_results[0].sam_string);

    // Do the comparison checks
    REQUIRE_FALSE(sam_line_from_read_common.empty());
    if (sam_line_from_read_common.back() == '\n') {
        sam_line_from_read_common.resize(sam_line_from_read_common.size() - 1);
    }

    const auto bam_fields = dorado::utils::split(sam_line_from_bam_ptr, '\t');
    const auto read_common_fields = dorado::utils::split(sam_line_from_read_common, '\t');
    REQUIRE(bam_fields.size() == read_common_fields.size());
    REQUIRE(bam_fields.size() >= 11);
    // first 11 mandatory fields should be identical
    for (std::size_t field_index{0}; field_index < 11; ++field_index) {
        CAPTURE(field_index);
        CAPTURE(sam_line_from_bam_ptr);
        CAPTURE(sam_line_from_read_common);
        CHECK(bam_fields[field_index] == read_common_fields[field_index]);
    }

    const auto bam_tags = get_tags_from_sam_line_fields(bam_fields);
    const auto read_common_tags = get_tags_from_sam_line_fields(read_common_fields);
    CHECK(bam_tags.size() == read_common_tags.size());
    for (const auto& [key, bam_value] : bam_tags) {
        CAPTURE(key);
        auto tag_entry = read_common_tags.find(key);
        REQUIRE(tag_entry != read_common_tags.end());
        // de:f tag compare to 4dp as this is the precision the minimap sam line generation function uses
        const auto& read_common_value = tag_entry->second;
        if (key == "de:f") {
            auto bam_value_as_float = std::stof(bam_value);
            auto read_common_value_as_float = std::stof(read_common_value);
            float diff = bam_value_as_float - read_common_value_as_float;
            constexpr float TOLERANCE_DP4{1e-04f};
            CHECK(diff < TOLERANCE_DP4);
            continue;
        }

        CHECK(read_common_value == bam_value);
    }
}

TEST_CASE_METHOD(AlignerNodeTestFixture,
                 "AlignerTest: Check SA tag in non-primary alignments has correct CIGAR string",
                 TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_basecall_target.fa";
    auto query = aligner_test_dir / "basecall_target.fa";

    std::string mm_options{"-k 15 -w 15"};
    bool soft_clipping = GENERATE(true, false);
    if (soft_clipping) {
        mm_options += " -Y";
    }
    auto options = dorado::alignment::mm2::parse_options(mm_options);

    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 1);
    REQUIRE(bam_records.size() == 3);

    bam1_t* primary_rec = bam_records[0].get();
    bam1_t* secondary_rec = bam_records[1].get();
    bam1_t* supplementary_rec = bam_records[2].get();

    // Check aux tags.
    CHECK_THAT(bam_aux2Z(bam_aux_get(primary_rec, "SA")), Equals("read2,1,+,999S899M,60,0;"));
    if (soft_clipping) {
        CHECK_THAT(bam_aux2Z(bam_aux_get(secondary_rec, "SA")),
                   Equals("read3,1,+,999M899S,0,0;read2,1,+,999S899M,60,0;"));
    } else {
        CHECK(bam_aux_get(secondary_rec, "SA") == nullptr);
    }
    CHECK_THAT(bam_aux2Z(bam_aux_get(supplementary_rec, "SA")), Equals("read3,1,+,999M899S,0,0;"));
}
