#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/HtsReader.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_GROUP "[bam_utils][aligner]"

namespace fs = std::filesystem;

namespace {

template <class... Args>
std::vector<dorado::BamPtr> RunAlignmentPipeline(dorado::HtsReader& reader, Args&&... args) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto aligner = pipeline_desc.add_node<dorado::Aligner>({sink}, args...);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));
    reader.read(*pipeline, 100);
    pipeline.reset();
    return ConvertMessages<dorado::BamPtr>(messages);
}

}  // namespace

TEST_CASE("AlignerTest: Check standard alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "target.fq";

    dorado::HtsReader reader(query.string());
    auto bam_records = RunAlignmentPipeline(reader, ref.string(), 15, 15, 1e9, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();
    bam1_t* in_rec = reader.record.get();

    // Check input/output reads are matching.
    std::string orig_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(in_rec), in_rec->core.l_qseq);
    std::string aligned_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(rec), rec->core.l_qseq);

    // Check quals are matching.
    std::vector<uint8_t> orig_qual(bam_get_qual(in_rec),
                                   bam_get_qual(in_rec) + in_rec->core.l_qseq);
    std::vector<uint8_t> aligned_qual(bam_get_qual(rec), bam_get_qual(rec) + rec->core.l_qseq);
    CHECK(orig_qual == aligned_qual);

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"NMi", "msi", "ASi", "nni", "def", "tpA", "cmi", "s1i", "rli"};
    for (auto tag : tags) {
        CHECK_THAT(aux, Contains(tag));
    }
}

TEST_CASE("AlignerTest: Check supplementary alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_aln_target.fa";
    auto query = aligner_test_dir / "supplementary_aln_query.fa";

    dorado::HtsReader reader(query.string());
    auto bam_records = RunAlignmentPipeline(reader, ref.string(), 15, 15, 1e9, 10);
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

TEST_CASE("AlignerTest: Check reverse complement alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "rev_target.fq";

    dorado::HtsReader reader(query.string());
    auto bam_records = RunAlignmentPipeline(reader, ref.string(), 15, 15, 1e9, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();
    bam1_t* in_rec = reader.record.get();

    // Check flag.
    CHECK(rec->core.flag & 0x10);

    // Check read reverse complementing.
    std::string orig_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(in_rec), in_rec->core.l_qseq);
    std::string aligned_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(rec), rec->core.l_qseq);
    CHECK(orig_read == dorado::utils::reverse_complement(aligned_read));

    // Check qual reversal.
    std::vector<uint8_t> orig_qual(bam_get_qual(in_rec),
                                   bam_get_qual(in_rec) + in_rec->core.l_qseq);
    std::vector<uint8_t> aligned_qual(bam_get_qual(rec), bam_get_qual(rec) + rec->core.l_qseq);
    std::reverse(aligned_qual.begin(), aligned_qual.end());
    CHECK(orig_qual == aligned_qual);
}

TEST_CASE("AlignerTest: Check dorado tags are retained", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "basecall_target.fa";
    auto query = aligner_test_dir / "basecall.sam";

    dorado::HtsReader reader(query.string());
    auto bam_records = RunAlignmentPipeline(reader, ref.string(), 15, 15, 1e9, 10);
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0].get();

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"RGZ", "MMZ", "MLB"};
    for (auto tag : tags) {
        CHECK_THAT(aux, Contains(tag));
    }
}

TEST_CASE("AlignerTest: Verify impact of updated aligner args", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "query.fa";

    // Run alignment with one set of k/w.
    {
        dorado::HtsReader reader(query.string());
        auto bam_records = RunAlignmentPipeline(reader, ref.string(), 28, 28, 1e9, 2);
        CHECK(bam_records.size() == 2);  // Generates 2 alignments.
    }

    // Run alignment with another set of k/w.
    {
        dorado::HtsReader reader(query.string());
        auto bam_records = RunAlignmentPipeline(reader, ref.string(), 5, 5, 1e9, 2);
        CHECK(bam_records.size() == 1);  // Generates 1 alignment.
    }
}

TEST_CASE("AlignerTest: Check Aligner crashes if multi index encountered", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "long_target.fa";

    CHECK_THROWS(dorado::Aligner(ref.string(), 5, 5, 1e3, 1));
}
