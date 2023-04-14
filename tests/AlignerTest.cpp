#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][aligner]"

namespace fs = std::filesystem;

TEST_CASE("AlignerTest: Check proper tag generation", TEST_GROUP) {
    using Catch::Matchers::Contains;

    fs::path aligner_test_dir = fs::path(get_aligner_data_dir());
    auto ref = aligner_test_dir / "target.fa";
    auto query = aligner_test_dir / "target.fa";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 10);
    dorado::utils::BamReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 1);

    bam1_t* rec = bam_records[0];

    // Check aux tags.
    uint32_t l_aux = bam_get_l_aux(rec);
    std::string aux((char*)bam_get_aux(rec), (char*)(bam_get_aux(rec) + l_aux));
    std::string tags[] = {"NMi", "msi", "ASi", "nni", "def", "tpA", "cmi", "s1i"};
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
    dorado::utils::Aligner aligner(sink, ref.string(), 10);
    dorado::utils::BamReader reader(query.string());
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
    auto ref = aligner_test_dir / "target.fa";
    auto query = aligner_test_dir / "rev_target.fa";

    // Run alignment.
    MessageSinkToVector<bam1_t*> sink(100);
    dorado::utils::Aligner aligner(sink, ref.string(), 10);
    dorado::utils::BamReader reader(query.string());
    reader.read(aligner, 100);
    auto bam_records = sink.get_messages();
    REQUIRE(bam_records.size() == 1);

    // Check first alignment is primary.
    bam1_t* rec = bam_records[0];

    // Check flag.
    REQUIRE(rec->core.flag & 0x10);
}
