#include "read_pipeline/nodes/WriterNode.h"

#include "TestUtils.h"
#include "catch2/catch_message.hpp"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "read_pipeline/base/HtsReader.h"
#include "read_pipeline/base/ReadPipeline.h"
#include "read_pipeline/base/flush_options.h"
#include "read_pipeline/base/messages.h"
#include "utils/stats.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <htslib/sam.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#define TEST_GROUP "[bam_utils][hts_writer][writer_node]"

namespace fs = std::filesystem;
using namespace dorado;
using Catch::Matchers::Equals;
using utils::HtsFile;

namespace {

constexpr std::string_view GPU_NAMES = "gpu_names:all";

// Node to mutate reads in the pipeline
// Updates the subread id for reads with a parent id that aren't duplex
class SubreadIdTaggerNode : public MessageSink {
public:
    SubreadIdTaggerNode() : MessageSink(100, 1) {}
    ~SubreadIdTaggerNode() { stop_input_processing(); }

    std::string get_name() const override { return "SubreadIdTagger"; }
    void terminate(const FlushOptions &) override { stop_input_processing(); };
    void restart() override {
        start_input_processing([this] { input_thread_fn(); }, "subreadidtagger_node");
    }

private:
    void input_thread_fn() {
        Message message;
        std::unordered_map<std::string, size_t> read_id_counts;
        while (get_input_message(message)) {
            auto bam_message = std::get<BamMessage>(std::move(message));

            int64_t dx_tag = 0;
            auto tag_str = bam_aux_get(bam_message.data.bam_ptr.get(), "dx");
            if (tag_str) {
                dx_tag = bam_aux2i(tag_str);
            }

            if (dx_tag != 1) {
                auto pid_tag = bam_aux_get(bam_message.data.bam_ptr.get(), "pi");
                if (pid_tag) {
                    std::string read_id = bam_aux2Z(pid_tag);
                    bam_message.data.subread_id = read_id_counts[read_id]++;
                }
            }
            send_message_to_sink(std::move(bam_message));
        }
    }
};

class HtsFileWriterTestsFixture {
public:
    HtsFileWriterTestsFixture()
            : m_out_path(tests::make_temp_dir("hts_writer_output")),
              m_out_bam(m_out_path.m_path / "out.bam"),
              m_in_sam(fs::path(get_data_dir("bam_reader") / "small.sam")) {
        std::filesystem::create_directories(m_out_path.m_path);
    }

protected:
    void generate_bam(HtsFile::OutputMode mode, int num_threads) {
        HtsReader reader(m_in_sam.string(), std::nullopt);

        std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
        {
            auto progress_cb = utils::ProgressCallback([](size_t) {});
            auto description_cb = utils::DescriptionCallback([](const std::string &) {});

            bool emit_fastq = mode == hts_writer::OutputMode::FASTQ;
            bool emit_sam = mode == hts_writer::OutputMode::SAM;

            auto hts_writer_builder = hts_writer::HtsFileWriterBuilder(
                    emit_fastq, emit_sam, false, m_out_path.m_path, num_threads, progress_cb,
                    description_cb, GPU_NAMES);

            std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
            CATCH_CHECK_FALSE(hts_file_writer == nullptr);
            CATCH_CHECK(to_string(hts_file_writer->get_mode()) == to_string(mode));

            writers.push_back(std::move(hts_file_writer));
        }

        PipelineDescriptor pipeline_desc;
        auto writer = pipeline_desc.add_node<WriterNode>({}, std::move(writers));

        pipeline_desc.add_node<SubreadIdTaggerNode>({writer});
        auto pipeline = Pipeline::create(std::move(pipeline_desc), nullptr);

        const auto &writer_ref = pipeline->get_node_ref<WriterNode>(writer);
        SamHdrPtr hdr(sam_hdr_dup(reader.header()));
        writer_ref.set_hts_file_header(std::move(hdr));

        reader.read(*pipeline, 1000);
        pipeline->terminate(DefaultFlushOptions());

        stats = writer_ref.sample_stats();
    }

    stats::NamedStats stats;

private:
    TempDir m_out_path;
    fs::path m_out_bam;
    fs::path m_in_sam;
};

std::string get_output_file(const std::filesystem::path &output_dir,
                            hts_writer::OutputMode output_mode) {
    const auto suffix = get_suffix(output_mode);
    std::optional<std::string> found;
    for (const auto &entry : std::filesystem::directory_iterator(
                 output_dir, std::filesystem::directory_options::skip_permission_denied)) {
        if (entry.path().extension().string() != suffix) {
            continue;
        }
        if (found.has_value()) {
            throw std::runtime_error(
                    "Expected temp_directory to contain only one output file with suffix: '" +
                    suffix + "'");
        } else {
            found = entry.path().string();
        }
    }
    if (found.has_value()) {
        return found.value();
    }
    throw std::runtime_error("Expected temp_directory to contain output file with suffix: '" +
                             suffix + "'");
};

}  // namespace

namespace dorado::hts_writer::test {

CATCH_TEST_CASE_METHOD(HtsFileWriterTestsFixture, "HtsFileWriterTest: Write BAM", TEST_GROUP) {
    int num_threads = GENERATE(1, 10);
    HtsFile::OutputMode output_mode = GENERATE(HtsFile::OutputMode::SAM, HtsFile::OutputMode::BAM,
                                               HtsFile::OutputMode::FASTQ);
    CATCH_CAPTURE(num_threads, to_string(output_mode));
    CATCH_CHECK_NOTHROW(generate_bam(output_mode, num_threads));
}

CATCH_TEST_CASE_METHOD(HtsFileWriterTestsFixture,
                       "HtsFileWriterTest: Count reads written",
                       TEST_GROUP) {
    CATCH_CHECK_NOTHROW(generate_bam(HtsFile::OutputMode::BAM, 1));

    std::vector<std::string> keys;
    keys.reserve(stats.size());
    std::transform(stats.begin(), stats.end(), std::back_inserter(keys),
                   [](const auto &pair) { return pair.first; });
    CATCH_CAPTURE(keys);

    CATCH_CHECK(stats.at("HtsFileWriter.unique_simplex_reads_written") == 6);
    CATCH_CHECK(stats.at("HtsFileWriter.split_reads_written") == 2);
}

CATCH_TEST_CASE("HtsFileWriterTest: Read and write FASTQ with tag", TEST_GROUP) {
    fs::path bam_test_dir = fs::path(get_data_dir("bam_reader"));
    std::string input_fastq_name = GENERATE("fastq_with_tags.fq", "fastq_with_us_and_tags.fq");
    auto input_fastq = bam_test_dir / input_fastq_name;
    auto tmp_dir = make_temp_dir("writer_test");

    // Read input file to check all tags are reads.
    HtsReader reader(input_fastq.string(), std::nullopt);

    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        auto progress_cb = utils::ProgressCallback([](size_t) {});
        auto description_cb = utils::DescriptionCallback([](const std::string &) {});

        auto hts_writer_builder = hts_writer::HtsFileWriterBuilder(
                true, false, false, tmp_dir.m_path, 1, progress_cb, description_cb, GPU_NAMES);

        std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
        CATCH_CHECK_FALSE(hts_file_writer == nullptr);
        CATCH_CHECK(hts_file_writer->get_mode() == OutputMode::FASTQ);

        SamHdrSharedPtr header(sam_hdr_init());
        hts_file_writer->set_header(header);

        writers.push_back(std::move(hts_file_writer));
    }
    WriterNode writer(std::move(writers));

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

    BamPtr bam_ptr{bam_dup1(reader.record.get())};
    BamMessage bam_message{HtsData{std::move(bam_ptr), "kit2"}, nullptr};

    writer.restart();
    writer.push_message(std::move(bam_message));
    writer.terminate(DefaultFlushOptions());

    const auto fastq_path = get_output_file(tmp_dir.m_path, OutputMode::FASTQ);
    CATCH_CAPTURE(fastq_path);

    // Read temporary file to make sure tags were correctly set.
    HtsReader new_fastq_reader(fastq_path, std::nullopt);
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
        std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
        {
            auto progress_cb = utils::ProgressCallback([](size_t) {});
            auto description_cb = utils::DescriptionCallback([](const std::string &) {});
            auto hts_writer_builder = hts_writer::HtsFileWriterBuilder(
                    false, true, false, tmp_dir.m_path, 1, progress_cb, description_cb, GPU_NAMES);

            std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
            CATCH_CHECK_FALSE(hts_file_writer == nullptr);
            CATCH_CHECK(hts_file_writer->get_mode() == OutputMode::SAM);

            SamHdrSharedPtr header(sam_hdr_init());
            hts_file_writer->set_header(header);

            writers.push_back(std::move(hts_file_writer));
        }
        WriterNode writer(std::move(writers));
        // Write into temporary folder.

        BamPtr bam_ptr{bam_dup1(fastq_reader.record.get())};
        BamMessage bam_message{HtsData{std::move(bam_ptr), "kit2"}, nullptr};

        writer.restart();
        writer.push_message(std::move(bam_message));
        writer.terminate(DefaultFlushOptions());
    }

    const auto out_sam = get_output_file(tmp_dir.m_path, OutputMode::SAM);

    // Read temporary file to make sure the 'fq' tag was not output.
    HtsReader new_fastq_reader(out_sam, std::nullopt);
    new_fastq_reader.read();

    fq_tag = bam_aux_get(new_fastq_reader.record.get(), "fq");
    CATCH_REQUIRE(fq_tag == nullptr);
}

}  // namespace dorado::hts_writer::test
