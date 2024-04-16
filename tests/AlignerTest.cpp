#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "alignment/Minimap2Aligner.h"
#include "fake_client_info.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "utils/PostCondition.h"
#include "utils/bam_utils.h"
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
            std::string reference_file,
            std::string bed_file,
            dorado::alignment::Minimap2Options options,
            int threads) {
        auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
        create_pipeline(index_file_access, reference_file, bed_file, options, threads);

        auto client_info = std::make_shared<dorado::DefaultClientInfo>();
        auto alignment_info = std::make_shared<dorado::AlignmentInfo>();
        alignment_info->minimap_options = options;
        alignment_info->reference_file = reference_file;
        client_info->set_alignment_info(alignment_info);
        reader.set_client_info(client_info);
        reader.read(*pipeline, 100);
        pipeline->terminate({});
        auto bam_messages = ConvertMessages<dorado::BamMessage>(std::move(m_output_messages));
        std::vector<dorado::BamPtr> result{};
        for (auto& bam_message : bam_messages) {
            result.push_back(std::move(bam_message.bam_ptr));
        }
        return result;
    }

    template <class MessageType, class MessageTypePtr = std::unique_ptr<MessageType>>
    MessageTypePtr RunPipelineForRead(const dorado::AlignmentInfo& loaded_align_info,
                                      const dorado::AlignmentInfo& client_align_info,
                                      std::string read_id,
                                      std::string sequence) {
        auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
        CHECK(index_file_access->load_index(loaded_align_info.reference_file,
                                            loaded_align_info.minimap_options,
                                            2) == dorado::alignment::IndexLoadResult::success);
        create_pipeline(index_file_access, 2);

        dorado::ReadCommon read_common{};
        auto client_info = std::make_shared<dorado::FakeClientInfo>();
        client_info->set_alignment_info(client_align_info);
        read_common.client_info = client_info;
        read_common.read_id = std::move(read_id);
        read_common.seq = std::move(sequence);

        auto read = std::make_unique<MessageType>();
        read->read_common = std::move(read_common);

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

        auto sam_line_buffer = dorado::utils::allocate_kstring();
        auto free_buffer =
                dorado::utils::PostCondition([&sam_line_buffer] { ks_free(&sam_line_buffer); });
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
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
    auto bh_tag_type = *(char*)bh_tag_ptr;
    CHECK(bh_tag_type == 'i');
    int32_t bh_tag_value = 0;
    memcpy(&bh_tag_value, bh_tag_ptr + 1, 4);
    CHECK(bh_tag_value == 3);
}

TEST_CASE_METHOD(AlignerNodeTestFixture, "AlignerTest: Check supplementary alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_aln_target.fa";
    auto query = aligner_test_dir / "supplementary_aln_query.fa";

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
    options.soft_clipping = GENERATE(true, false);
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 10);
    REQUIRE(bam_records.size() == 3);

    bam1_t* primary_rec = bam_records[0].get();
    bam1_t* secondary_rec = bam_records[1].get();
    bam1_t* supplementary_rec = bam_records[2].get();

    // Check aux tags.
    if (options.soft_clipping) {
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
        auto options = dorado::alignment::dflt_options;
        options.kmer_size = options.window_size = 28;
        options.index_batch_size = 1'000'000'000ull;
        dorado::HtsReader reader(query.string(), std::nullopt);
        auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 2);
        CHECK(bam_records.size() == 2);  // Generates 2 alignments.
    }

    // Run alignment with another set of k/w.
    {
        auto options = dorado::alignment::dflt_options;
        options.kmer_size = options.window_size = 5;
        options.index_batch_size = 1'000'000'000ull;
        dorado::HtsReader reader(query.string(), std::nullopt);
        auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 2);
        CHECK(bam_records.size() == 1);  // Generates 1 alignment.
    }
}

TEST_CASE("AlignerTest: Check AlignerNode crashes if multi index encountered", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "long_target.fa";

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 5;
    options.index_batch_size = 1000ull;
    auto index_file_access = std::make_shared<dorado::alignment::IndexFileAccess>();
    CHECK_THROWS(dorado::AlignerNode(index_file_access, ref.string(), "", options, 1));
}

SCENARIO_METHOD(AlignerNodeTestFixture, "AlignerNode push SimplexRead", TEST_GROUP) {
    GIVEN("AlgnerNode constructed with populated index file collection") {
        const std::string READ_ID{"aligner_node_test"};
        fs::path aligner_test_dir{get_aligner_data_dir()};
        auto ref = aligner_test_dir / "target.fq";

        dorado::AlignmentInfo align_info{};
        align_info.minimap_options = dorado::alignment::dflt_options;
        align_info.reference_file = ref.string();

        const std::string TEST_SEQUENCE{"ACGTACGTACGTACGT"};

        AND_GIVEN("client with no alignment requirements") {
            const dorado::AlignmentInfo EMPTY_ALIGN_INFO{};
            WHEN("push simplex read to pipeline") {
                auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(
                        align_info, EMPTY_ALIGN_INFO, READ_ID, TEST_SEQUENCE);

                THEN("Output simplex read has empty alignment_string") {
                    REQUIRE(simplex_read->read_common.alignment_string.empty());
                }
            }

            WHEN("push duplex read to pipeline") {
                auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(
                        align_info, EMPTY_ALIGN_INFO, READ_ID, TEST_SEQUENCE);
                THEN("Output duplex read has empty alignment_string") {
                    REQUIRE(duplex_read->read_common.alignment_string.empty());
                }
            }
        }

        AND_GIVEN("client requiring alignment") {
            WHEN("push simplex read with no alignment matches to pipeline") {
                auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(align_info, align_info,
                                                                            READ_ID, TEST_SEQUENCE);

                THEN("Output simplex read has alignment_string populated") {
                    REQUIRE_FALSE(simplex_read->read_common.alignment_string.empty());
                }

                THEN("Output simplex read has alignment_string containing unmapped sam line") {
                    const std::string expected{READ_ID +
                                               dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                    REQUIRE(simplex_read->read_common.alignment_string == expected);
                }
            }

            WHEN("push duplex read with no alignment matches to pipeline") {
                auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(align_info, align_info,
                                                                          READ_ID, TEST_SEQUENCE);

                THEN("Output duplex read has alignment_string populated") {
                    REQUIRE_FALSE(duplex_read->read_common.alignment_string.empty());
                }

                THEN("Output duplex read has alignment_string containing unmapped sam line") {
                    const std::string expected{READ_ID +
                                               dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                    REQUIRE(duplex_read->read_common.alignment_string == expected);
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
                        REQUIRE(simplex_read->read_common.alignment_string.substr(
                                        0, READ_ID.size()) == READ_ID);
                    }

                    THEN("Output sam line contains sequence string") {
                        REQUIRE_FALSE(simplex_read->read_common.alignment_string.find(sequence) ==
                                      std::string::npos);
                    }
                }

                WHEN("pushed as duplex read to pipeline") {
                    auto duplex_read = RunPipelineForRead<dorado::DuplexRead>(
                            align_info, align_info, READ_ID, sequence);

                    THEN("Output sam line has read_id as QNAME") {
                        const std::string expected{READ_ID +
                                                   dorado::alignment::UNMAPPED_SAM_LINE_STRIPPED};
                        REQUIRE(duplex_read->read_common.alignment_string.substr(
                                        0, READ_ID.size()) == READ_ID);
                    }

                    THEN("Output sam line contains sequence string") {
                        REQUIRE_FALSE(duplex_read->read_common.alignment_string.find(sequence) ==
                                      std::string::npos);
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 5;
    options.index_batch_size = 1'000'000'000ull;

    // Get the sam line from BAM pipeline
    dorado::HtsReader bam_reader(query, std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(bam_reader, ref, "", options, 2);
    CHECK(bam_records.size() == 1);
    auto sam_line_from_bam_ptr = get_sam_line_from_bam(std::move(bam_records[0]));

    // Get the sam line from ReadCommon pipeline
    auto [read_id, sequence] = get_read_id_and_sequence_from_fasta(query);
    dorado::AlignmentInfo align_info{};
    align_info.minimap_options = options;
    align_info.reference_file = ref;
    auto simplex_read = RunPipelineForRead<dorado::SimplexRead>(
            align_info, align_info, std::move(read_id), std::move(sequence));
    auto sam_line_from_read_common = std::move(simplex_read->read_common.alignment_string);

    // Do the comparison checks
    CHECK_FALSE(sam_line_from_read_common.empty());

    if (sam_line_from_read_common.at(sam_line_from_read_common.size() - 1) == '\n') {
        sam_line_from_read_common =
                sam_line_from_read_common.substr(0, sam_line_from_read_common.size() - 1);
    }

    const auto bam_fields = dorado::utils::split(sam_line_from_bam_ptr, '\t');
    const auto read_common_fields = dorado::utils::split(sam_line_from_read_common, '\t');
    CHECK(bam_fields.size() == read_common_fields.size());
    CHECK(bam_fields.size() >= 11);
    // first 11 mandatory fields should be identical
    for (std::size_t field_index{0}; field_index < 11; ++field_index) {
        CHECK(bam_fields[field_index] == read_common_fields[field_index]);
    }

    const auto bam_tags = get_tags_from_sam_line_fields(bam_fields);
    const auto read_common_tags = get_tags_from_sam_line_fields(read_common_fields);
    CHECK(bam_tags.size() == read_common_tags.size());
    for (const auto& [key, bam_value] : bam_tags) {
        INFO(key);
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

    auto options = dorado::alignment::dflt_options;
    options.kmer_size = options.window_size = 15;
    options.index_batch_size = 1'000'000'000ull;
    options.soft_clipping = GENERATE(true, false);
    dorado::HtsReader reader(query.string(), std::nullopt);
    auto bam_records = RunPipelineWithBamMessages(reader, ref.string(), "", options, 1);
    REQUIRE(bam_records.size() == 3);

    bam1_t* primary_rec = bam_records[0].get();
    bam1_t* secondary_rec = bam_records[1].get();
    bam1_t* supplementary_rec = bam_records[2].get();

    // Check aux tags.
    CHECK_THAT(bam_aux2Z(bam_aux_get(primary_rec, "SA")), Equals("read2,1,+,999S899M,60,0;"));
    if (options.soft_clipping) {
        CHECK_THAT(bam_aux2Z(bam_aux_get(secondary_rec, "SA")),
                   Equals("read3,1,+,999M899S,0,0;read2,1,+,999S899M,60,0;"));
    } else {
        CHECK(bam_aux_get(secondary_rec, "SA") == nullptr);
    }
    CHECK_THAT(bam_aux2Z(bam_aux_get(supplementary_rec, "SA")), Equals("read3,1,+,999M899S,0,0;"));
}
