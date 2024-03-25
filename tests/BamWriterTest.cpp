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
using Catch::Matchers::Equals;
using utils::HtsFile;

class HtsWriterTestsFixture {
public:
    HtsWriterTestsFixture()
            : m_out_path(TempDir(fs::temp_directory_path() / "hts_writer_output")),
              m_out_bam(m_out_path.m_path / "out.bam"),
              m_in_sam(fs::path(get_data_dir("bam_reader") / "small.sam")) {
        std::filesystem::create_directories(m_out_path.m_path);
    }

protected:
    void generate_bam(HtsFile::OutputMode mode, int num_threads) {
        HtsReader reader(m_in_sam.string(), std::nullopt);

        utils::HtsFile hts_file(m_out_bam.string(), mode, num_threads);
        hts_file.set_and_write_header(reader.header);

        PipelineDescriptor pipeline_desc;
        auto writer = pipeline_desc.add_node<HtsWriter>({}, hts_file);
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);

        reader.read(*pipeline, 1000);
        pipeline->terminate(DefaultFlushOptions());

        auto& writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(writer));
        stats = writer_ref.sample_stats();

        hts_file.finalise([](size_t) { /* noop */ }, num_threads);
    }

    stats::NamedStats stats;

private:
    TempDir m_out_path;
    fs::path m_out_bam;
    fs::path m_in_sam;
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

TEST_CASE("HtsWriterTest: Read and write FASTQ with tag", TEST_GROUP) {
    fs::path bam_test_dir = fs::path(get_data_dir("bam_reader"));
    auto input_fastq = bam_test_dir / "fastq_with_tags.fq";
    auto tmp_dir = TempDir(fs::temp_directory_path() / "writer_test");
    std::filesystem::create_directories(tmp_dir.m_path);
    auto out_fastq = tmp_dir.m_path / "output.fq";

    // Read input file to check all tags are reads.
    HtsReader reader(input_fastq.string(), std::nullopt);
    {
        // Write with tags into temporary folder.
        utils::HtsFile hts_file(out_fastq.string(), HtsFile::OutputMode::FASTQ, 2);
        HtsWriter writer(hts_file);
        reader.read();
        CHECK_THAT(bam_aux2Z(bam_aux_get(reader.record.get(), "RG")),
                   Equals("6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4."
                          "3.0"));
        CHECK_THAT(bam_aux2Z(bam_aux_get(reader.record.get(), "st")),
                   Equals("2023-06-22T07:17:48.308+00:00"));
        writer.write(reader.record.get());
        hts_file.finalise([](size_t) { /* noop */ }, 2);
    }

    // Read temporary file to make sure tags were correctly set.
    HtsReader new_fastq_reader(out_fastq.string(), std::nullopt);
    new_fastq_reader.read();
    CHECK_THAT(
            bam_aux2Z(bam_aux_get(new_fastq_reader.record.get(), "RG")),
            Equals("6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4.3.0"));
    CHECK_THAT(bam_aux2Z(bam_aux_get(new_fastq_reader.record.get(), "st")),
               Equals("2023-06-22T07:17:48.308+00:00"));
}
