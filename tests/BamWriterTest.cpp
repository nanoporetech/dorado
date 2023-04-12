#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "htslib/sam.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][bam_reader]"

namespace fs = std::filesystem;

TEMPLATE_TEST_CASE_SIG("BamWriterTest: Write BAM", TEST_GROUP, ((int threads), threads), 1, 10) {
    fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
    auto sam = aligner_test_dir / "small.sam";
    auto bam_path = fs::temp_directory_path() / "out.bam";

    {
        // Running within a local scope to ensure file paths are closed
        // before removing the file.
        dorado::utils::BamReader reader(sam.string());
        dorado::utils::BamWriter writer(bam_path.string(), threads);

        dorado::utils::sq_t sequences;
        CHECK(sequences.size() == 0);  // No sequence information for this test.
        writer.write_header(reader.m_header, sequences);
        reader.read(writer, 1000);

        writer.join();
    }

    // Test only checks that no API calls raised any exceptions.
    REQUIRE(true);
    CHECK(fs::remove(bam_path));
}
