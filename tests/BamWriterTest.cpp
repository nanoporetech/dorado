#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][bam_reader]"

namespace fs = std::filesystem;

TEST_CASE("BamWriterTest: Write BAM", TEST_GROUP) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";

    dorado::utils::BamReader reader(sam.string());
    dorado::utils::BamWriter writer("-");

    dorado::utils::sq_t sequences;
    CHECK(sequences.size() == 0);  // No sequence information for this test.
    writer.write_header(reader.m_header, sequences);

    while (reader.read()) {
        writer.push_message(reader.m_record);
    }
    writer.terminate();
    writer.join();

    // Test only checks that no API calls raised any exceptions.
    REQUIRE(true);
}
