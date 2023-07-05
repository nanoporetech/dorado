#include "TestUtils.h"
#include "htslib/sam.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "utils/bam_utils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[bam_utils][hts_writer]"

namespace fs = std::filesystem;
using namespace dorado;

class HtsWriterTestsFixture {
public:
    HtsWriterTestsFixture() {
        fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
        m_in_sam = aligner_test_dir / "small.sam";
        m_out_bam = fs::temp_directory_path() / "out.bam";
    }

    ~HtsWriterTestsFixture() { fs::remove(m_out_bam); }

protected:
    void generate_bam(HtsWriter::OutputMode mode, int num_threads) {
        HtsReader reader(m_in_sam.string());
        PipelineDescriptor pipeline_desc;
        auto writer =
                pipeline_desc.add_node<HtsWriter>({}, m_out_bam.string(), mode, num_threads, 0);
        auto pipeline = Pipeline::create(std::move(pipeline_desc));

        auto& writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(writer));
        writer_ref.set_and_write_header(reader.header);

        reader.read(*pipeline, 1000);
        pipeline.reset();
    }

private:
    fs::path m_in_sam;
    fs::path m_out_bam;
};

TEST_CASE_METHOD(HtsWriterTestsFixture, "HtsWriterTest: Write BAM", TEST_GROUP) {
    int num_threads = GENERATE(1, 10);
    HtsWriter::OutputMode emit_fastq = GENERATE(
            HtsWriter::OutputMode::SAM, HtsWriter::OutputMode::BAM, HtsWriter::OutputMode::FASTQ);
    REQUIRE_NOTHROW(generate_bam(emit_fastq, num_threads));
}

TEST_CASE("HtsWriterTest: Output mode conversion", TEST_GROUP) {
    CHECK(HtsWriter::get_output_mode("sam") == HtsWriter::OutputMode::SAM);
    CHECK(HtsWriter::get_output_mode("bam") == HtsWriter::OutputMode::BAM);
    CHECK(HtsWriter::get_output_mode("fastq") == HtsWriter::OutputMode::FASTQ);
    CHECK_THROWS_WITH(HtsWriter::get_output_mode("blah"), "Unknown output mode: blah");
}
