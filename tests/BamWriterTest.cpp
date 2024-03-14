#include "TestUtils.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "utils/bam_utils.h"
#include "utils/hts_file.h"
#include "utils/stats.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>

#define TEST_GROUP "[bam_utils][hts_writer]"

namespace fs = std::filesystem;
using namespace dorado;
using utils::HtsFile;

class HtsWriterTestsFixture {
public:
    HtsWriterTestsFixture() {
        fs::path aligner_test_dir = fs::path(get_data_dir("bam_reader"));
        m_in_sam = aligner_test_dir / "small.sam";
        m_out_bam = fs::temp_directory_path() / "out.bam";
    }

    ~HtsWriterTestsFixture() { fs::remove(m_out_bam); }

protected:
    void generate_bam(HtsFile::OutputMode mode, int num_threads) {
        utils::HtsFile hts_file(m_out_bam.string(), mode, num_threads);

        HtsReader reader(m_in_sam.string(), std::nullopt);
        PipelineDescriptor pipeline_desc;
        auto writer = pipeline_desc.add_node<HtsWriter>({}, hts_file);
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);

        auto& writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(writer));
        writer_ref.set_and_write_header(reader.header);

        reader.read(*pipeline, 1000);
        pipeline->terminate(DefaultFlushOptions());
        hts_file.finalise();

        stats = writer_ref.sample_stats();
        pipeline.reset();
    }

    stats::NamedStats stats;

private:
    fs::path m_in_sam;
    fs::path m_out_bam;
};

TEST_CASE_METHOD(HtsWriterTestsFixture, "HtsWriterTest: Write BAM", TEST_GROUP) {
    int num_threads = GENERATE(1, 10);
    HtsFile::OutputMode emit_fastq = GENERATE(HtsFile::OutputMode::SAM, HtsFile::OutputMode::BAM,
                                              HtsFile::OutputMode::FASTQ);
    CAPTURE(num_threads);
    CAPTURE(emit_fastq);
    CHECK_NOTHROW(generate_bam(emit_fastq, num_threads));
}

TEST_CASE("HtsWriterTest: Output mode conversion", TEST_GROUP) {
    CHECK(HtsWriter::get_output_mode("sam") == HtsFile::OutputMode::SAM);
    CHECK(HtsWriter::get_output_mode("bam") == HtsFile::OutputMode::BAM);
    CHECK(HtsWriter::get_output_mode("fastq") == HtsFile::OutputMode::FASTQ);
    CHECK_THROWS_WITH(HtsWriter::get_output_mode("blah"), "Unknown output mode: blah");
}

TEST_CASE_METHOD(HtsWriterTestsFixture, "HtsWriter: Count reads written", TEST_GROUP) {
    CHECK_NOTHROW(generate_bam(HtsFile::OutputMode::BAM, 1));

    CHECK(stats.at("unique_simplex_reads_written") == 6);
    CHECK(stats.at("split_reads_written") == 2);
}
