#include "TestUtils.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ReadPipeline.h"
#include "utils/bam_utils.h"
#include "utils/hts_file.h"
#include "utils/stats.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <filesystem>

#define TEST_GROUP "[bam_utils][hts_writer]"

namespace fs = std::filesystem;
using namespace dorado;
using Catch::Matchers::Equals;
using utils::HtsFile;

namespace {
class HtsWriterTestsFixture {
public:
    HtsWriterTestsFixture()
            : m_out_path(tests::make_temp_dir("hts_writer_output")),
              m_out_bam(m_out_path.m_path / "out.bam"),
              m_in_sam(fs::path(get_data_dir("bam_reader") / "small.sam")) {
        std::filesystem::create_directories(m_out_path.m_path);
    }

protected:
    void generate_bam(HtsFile::OutputMode mode, int num_threads) {
        HtsReader reader(m_in_sam.string(), std::nullopt);

        utils::HtsFile hts_file(m_out_bam.string(), mode, num_threads, false);
        hts_file.set_header(reader.header());

        PipelineDescriptor pipeline_desc;
        auto writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, "");
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);

        reader.read(*pipeline, 1000);
        pipeline->terminate(DefaultFlushOptions());

        auto& writer_ref = pipeline->get_node_ref<HtsWriter>(writer);
        stats = writer_ref.sample_stats();

        hts_file.finalise([](size_t) { /* noop */ });
    }

    stats::NamedStats stats;

private:
    TempDir m_out_path;
    fs::path m_out_bam;
    fs::path m_in_sam;
};
}  // namespace

namespace dorado::hts_writer::test {

CATCH_TEST_CASE_METHOD(HtsWriterTestsFixture, "HtsWriterTest: Write BAM", TEST_GROUP) {
    int num_threads = GENERATE(1, 10);
    HtsFile::OutputMode emit_fastq = GENERATE(HtsFile::OutputMode::SAM, HtsFile::OutputMode::BAM,
                                              HtsFile::OutputMode::FASTQ);
    CATCH_CAPTURE(num_threads);
    CATCH_CAPTURE(emit_fastq);
    CATCH_CHECK_NOTHROW(generate_bam(emit_fastq, num_threads));
}

CATCH_TEST_CASE("HtsWriterTest: Output mode conversion", TEST_GROUP) {
    CATCH_CHECK(HtsWriter::get_output_mode("sam") == HtsFile::OutputMode::SAM);
    CATCH_CHECK(HtsWriter::get_output_mode("bam") == HtsFile::OutputMode::BAM);
    CATCH_CHECK(HtsWriter::get_output_mode("fastq") == HtsFile::OutputMode::FASTQ);
    CATCH_CHECK_THROWS_WITH(HtsWriter::get_output_mode("blah"), "Unknown output mode: blah");
}

CATCH_TEST_CASE_METHOD(HtsWriterTestsFixture, "HtsWriter: Count reads written", TEST_GROUP) {
    CATCH_CHECK_NOTHROW(generate_bam(HtsFile::OutputMode::BAM, 1));

    CATCH_CHECK(stats.at("unique_simplex_reads_written") == 6);
    CATCH_CHECK(stats.at("split_reads_written") == 2);
}

CATCH_TEST_CASE("HtsWriterTest: Read and write FASTQ with tag", TEST_GROUP) {
    fs::path bam_test_dir = fs::path(get_data_dir("bam_reader"));
    std::string input_fastq_name = GENERATE("fastq_with_tags.fq", "fastq_with_us_and_tags.fq");
    auto input_fastq = bam_test_dir / input_fastq_name;
    auto tmp_dir = make_temp_dir("writer_test");
    auto out_fastq = tmp_dir.m_path / "output.fq";

    // Read input file to check all tags are reads.
    HtsReader reader(input_fastq.string(), std::nullopt);
    {
        // Write with tags into temporary folder.
        utils::HtsFile hts_file(out_fastq.string(), HtsFile::OutputMode::FASTQ, 2, false);
        HtsWriter writer(hts_file, "");
        reader.read();
        const auto rg_tag = bam_aux_get(reader.record.get(), "RG");
        CATCH_REQUIRE(rg_tag != nullptr);
        CATCH_CHECK_THAT(
                bam_aux2Z(rg_tag),
                Equals("6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4."
                       "3.0"));
        const auto st_tag = bam_aux_get(reader.record.get(), "st");
        CATCH_REQUIRE(st_tag != nullptr);
        CATCH_CHECK_THAT(bam_aux2Z(st_tag), Equals("2023-06-22T07:17:48.308+00:00"));
        writer.write(reader.record.get());
        hts_file.finalise([](size_t) { /* noop */ });
    }

    // Read temporary file to make sure tags were correctly set.
    HtsReader new_fastq_reader(out_fastq.string(), std::nullopt);
    new_fastq_reader.read();
    CATCH_CHECK_THAT(
            bam_aux2Z(bam_aux_get(new_fastq_reader.record.get(), "RG")),
            Equals("6a94c5e38fbe36232d63fd05555e41368b204cda_dna_r10.4.1_e8.2_400bps_hac@v4.3.0"));
    CATCH_CHECK_THAT(bam_aux2Z(bam_aux_get(new_fastq_reader.record.get(), "st")),
                     Equals("2023-06-22T07:17:48.308+00:00"));
}

CATCH_TEST_CASE(
        "HtsWriterTest: Read fastq with minKNOW header does not write out the bam tag containing "
        "the input fastq header",
        TEST_GROUP) {
    const fs::path bam_test_dir = fs::path(get_data_dir("bam_reader"));
    const auto minknow_fastq_file = bam_test_dir / "fastq_with_minknow_header.fq";
    dorado::HtsReader fastq_reader(minknow_fastq_file.string(), std::nullopt);
    const auto tmp_dir = make_temp_dir("writer_test");
    const auto out_sam = tmp_dir.m_path / "output.sam";

    // Read the minkow style fastq and confirm the header line is written to the bam aux tag 'fq'
    CATCH_CHECK(fastq_reader.read());
    auto fq_tag = bam_aux_get(fastq_reader.record.get(), "fq");
    CATCH_REQUIRE(fq_tag != nullptr);
    const auto fq_tag_value = bam_aux2Z(fq_tag);
    CATCH_REQUIRE(fq_tag_value != nullptr);
    const std::string fq_header{fq_tag_value};
    CATCH_CHECK(fq_header ==
                "@c2707254-5445-4cfb-a414-fce1f12b56c0 "
                "runid=5c76f4079ee8f04e80b4b8b2c4b677bce7bebb1e "
                "read=1728 ch=332 start_time=2017-06-16T15:31:55Z");

    {
        // Write into temporary folder.
        utils::HtsFile hts_file(out_sam.string(), HtsFile::OutputMode::SAM, 2, false);
        hts_file.set_header(fastq_reader.header());
        HtsWriter writer(hts_file, "");
        writer.write(fastq_reader.record.get());
        hts_file.finalise([](size_t) { /* noop */ });
    }

    // Read temporary file to make sure the 'fq' tag was not output.
    HtsReader new_fastq_reader(out_sam.string(), std::nullopt);
    new_fastq_reader.read();

    fq_tag = bam_aux_get(new_fastq_reader.record.get(), "fq");
    CATCH_REQUIRE(fq_tag == nullptr);
}

}  // namespace dorado::hts_writer::test