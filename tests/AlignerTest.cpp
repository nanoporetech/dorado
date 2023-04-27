#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"
#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][aligner]"

namespace fs = std::filesystem;

TEST_CASE("AlignerTest: Check standard alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "target.fq";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 15, 15, 10);
    dorado::utils::HtsReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0];
    bam1_t* in_rec = reader.record;

    // Check input/output reads are matching.
    std::string orig_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(in_rec), in_rec->core.l_qseq);
    std::string aligned_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(rec), rec->core.l_qseq);
    REQUIRE(orig_read == aligned_read);

    // Check quals are matching.
    std::vector<uint8_t> orig_qual(bam_get_qual(in_rec),
                                   bam_get_qual(in_rec) + in_rec->core.l_qseq);
    std::vector<uint8_t> aligned_qual(bam_get_qual(rec), bam_get_qual(rec) + rec->core.l_qseq);
    REQUIRE(orig_qual == aligned_qual);

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"NMi", "msi", "ASi", "nni", "def", "tpA", "cmi", "s1i", "rli"};
    for (auto tag : tags) {
        REQUIRE_THAT(aux, Contains(tag));
    }
}

TEST_CASE("AlignerTest: Check supplementary alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "supplementary_aln_target.fa";
    auto query = aligner_test_dir / "supplementary_aln_query.fa";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 15, 15, 10);
    dorado::utils::HtsReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 2);

    // Check first alignment is primary.
    bam1_t* rec = bam_records[0];

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"tpAP"};
    for (auto tag : tags) {
        REQUIRE_THAT(aux, Contains(tag));
    }
    REQUIRE(rec->core.l_qseq > 0);  // Primary alignment should have SEQ.

    // Check second alignment is secondary.
    rec = bam_records[1];

    // Check aux tags.
    l_aux = bam_get_l_aux(rec);
    aux = std::string((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string sec_tags[] = {"tpAS"};
    for (auto tag : sec_tags) {
        REQUIRE_THAT(aux, Contains(tag));
    }
    REQUIRE(rec->core.l_qseq == 0);  // Secondary alignment doesn't need SEQ.
}

TEST_CASE("AlignerTest: Check reverse complement alignment", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "rev_target.fq";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 15, 15, 10);
    dorado::utils::HtsReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0];
    bam1_t* in_rec = reader.record;

    // Check flag.
    REQUIRE(rec->core.flag & 0x10);

    // Check read reverse complementing.
    std::string orig_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(in_rec), in_rec->core.l_qseq);
    std::string aligned_read =
            dorado::utils::convert_nt16_to_str(bam_get_seq(rec), rec->core.l_qseq);
    REQUIRE(orig_read == dorado::utils::reverse_complement(aligned_read));

    // Check qual reversal.
    std::vector<uint8_t> orig_qual(bam_get_qual(in_rec),
                                   bam_get_qual(in_rec) + in_rec->core.l_qseq);
    std::vector<uint8_t> aligned_qual(bam_get_qual(rec), bam_get_qual(rec) + rec->core.l_qseq);
    std::reverse(aligned_qual.begin(), aligned_qual.end());
    REQUIRE(orig_qual == aligned_qual);
}

TEST_CASE("AlignerTest: Check dorado tags are retained", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "basecall_target.fa";
    auto query = aligner_test_dir / "basecall.sam";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 15, 15, 10);
    dorado::utils::HtsReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0];

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"RGZ", "MMZ", "MLB"};
    for (auto tag : tags) {
        REQUIRE_THAT(aux, Contains(tag));
    }
}

TEST_CASE("AlignerTest: Verify impact of updated aligner args", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fq";
    auto query = aligner_test_dir / "query.fa";

    // Run alignment with one set of k/w.
    {
        MessageSinkToVector<bam1_t*> sink(100);
        dorado::utils::Aligner aligner(sink, ref.string(), 28, 28, 2);
        dorado::utils::HtsReader reader(query.string());
        reader.read(aligner, 100);
        auto bam_records = sink.get_messages();
        REQUIRE(bam_records.size() == 2);  // Generates 2 alignments.
    }

    // Run alignment with another set of k/w.
    {
        MessageSinkToVector<bam1_t*> sink(100);
        dorado::utils::Aligner aligner(sink, ref.string(), 5, 5, 2);
        dorado::utils::HtsReader reader(query.string());
        reader.read(aligner, 100);
        auto bam_records = sink.get_messages();
        REQUIRE(bam_records.size() == 1);  // Generates 1 alignment.
    }
}
