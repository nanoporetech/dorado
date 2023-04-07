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
    dorado::utils::BamReader reader(query);
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
