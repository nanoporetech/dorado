#include "TestUtils.h"
#include "hts_utils/header_utils.h"
#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "hts_writer/StreamHtsFileWriter.h"
#include "hts_writer/Structure.h"
#include "utils/PostCondition.h"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <htslib/sam.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define TEST_GROUP "[hts_file]"
static constexpr int NUM_THREADS = 4;

namespace fs = std::filesystem;
using namespace dorado;
using utils::HtsFile;

namespace {
struct Tester {
    dorado::tests::TempDir output_test_dir;
    fs::path file_in_path;
    fs::path file_out_path;
    std::vector<dorado::BamPtr> records;
    HtsFilePtr file_in;
    SamHdrPtr header_in, header_out;
    std::vector<size_t> indices;

    Tester()
            : output_test_dir(tests::make_temp_dir("hts_writer_output")),
              file_in_path(fs::path(get_data_dir("hts_file")) / "test_data.bam"),
              file_out_path(output_test_dir.m_path / "test_output.bam") {}

    void read_input_records() {
        // Read the test data into a vector of BAM records.
        file_in.reset(hts_open(file_in_path.string().c_str(), "r"));
        header_in.reset(sam_hdr_read(file_in.get()));
        BamPtr record(bam_init1());
        while (sam_read1(file_in.get(), header_in.get(), record.get()) >= 0) {
            records.emplace_back(std::move(record));
            record.reset(bam_init1());
        }
        file_in.reset();
        header_out.reset(sam_hdr_dup(header_in.get()));
        sam_hdr_change_HD(header_out.get(), "SO", "unknown");
        header_in.reset();
        indices.resize(records.size());
        std::iota(indices.begin(), indices.end(), 0);
        auto rng = std::default_random_engine{};
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    int write_output_records(size_t buffer_size) {
        bool sort_bam = (buffer_size > 0);
        HtsFile file_out(file_out_path.string(), HtsFile::OutputMode::BAM, NUM_THREADS, sort_bam);
        if (sort_bam) {
            file_out.set_buffer_size(buffer_size);
        }
        file_out.set_header(header_out.get());
        for (size_t i = 0; i < records.size(); ++i) {
            file_out.write(records[indices[i]].get());
        }
        int callback_calls = 0;
        auto callback = [&callback_calls](size_t) { ++callback_calls; };
        file_out.finalise(callback);
        return callback_calls;
    }

    void check_output(bool is_sorted) {
        file_in.reset(hts_open(file_out_path.string().c_str(), "r"));
        header_in.reset(sam_hdr_read(file_in.get()));
        BamPtr record(bam_init1());
        size_t index = 0;
        uint64_t last_sorting_key = 0;
        while (sam_read1(file_in.get(), header_in.get(), record.get()) >= 0) {
            CATCH_REQUIRE(index < records.size());
            if (is_sorted) {
                auto sorting_key = HtsFile::calculate_sorting_key(record.get());
                CATCH_REQUIRE(sorting_key >= last_sorting_key);
                last_sorting_key = sorting_key;
            } else {
                // Output records should be in the order they were written.
                auto expected_record = records[indices[index]].get();
                std::string qname(bam_get_qname(record.get()));
                std::string expected_qname(bam_get_qname(expected_record));
                CATCH_REQUIRE(qname == expected_qname);
            }
            ++index;
            record.reset(bam_init1());
        }
        file_in.reset();
        header_in.reset();
    }
};

std::vector<std::string> get_dummy_filenames(const std::string& base_name,
                                             const std::string& ext,
                                             size_t count,
                                             size_t first_index) {
    std::vector<std::string> files(count);
    for (size_t i = 0; i < count; ++i) {
        files[i] = base_name;
        files[i] += std::to_string(i + first_index);
        files[i] += ext;
    }
    return files;
}

std::string filepath(const std::string& folder, const std::string& filename) {
    return (std::filesystem::path(folder) / filename).string();
}

}  // namespace

CATCH_TEST_CASE("HtsFileTest: Write to unsorted file", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    int callback_calls = tester.write_output_records(0);
    CATCH_REQUIRE(callback_calls == 2);

    tester.check_output(false);
}

CATCH_TEST_CASE("HtsFileTest: Write to single sorted file", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    // A 5 MB buffer should make sure only a single temp file is written.
    int callback_calls = tester.write_output_records(5000000);
    CATCH_REQUIRE(callback_calls == 3);

    tester.check_output(true);
}

CATCH_TEST_CASE("HtsFileTest: Write to multiple sorted files, and merge", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    // A 200 KB buffer should make sure multiple temp files are written.
    int callback_calls = tester.write_output_records(200000);
    CATCH_REQUIRE(callback_calls > 4);

    tester.check_output(true);
}

CATCH_TEST_CASE("HtsFileTest: construct with zero threads for sorted BAM does not throw",
                TEST_GROUP) {
    Tester tester;
    std::unique_ptr<HtsFile> cut{};
    auto finalize_file = utils::PostCondition([&cut] { cut->finalise([](size_t) {}); });

    CATCH_REQUIRE_NOTHROW(cut = std::make_unique<HtsFile>(tester.file_out_path.string(),
                                                          HtsFile::OutputMode::BAM, 0, true));
}

CATCH_TEST_CASE("HtsFileTest: construct with zero threads for unsorted BAM does not throw",
                TEST_GROUP) {
    Tester tester;
    std::unique_ptr<HtsFile> cut{};
    auto finalize_file = utils::PostCondition([&cut] { cut->finalise([](size_t) {}); });

    CATCH_REQUIRE_NOTHROW(cut = std::make_unique<HtsFile>(tester.file_out_path.string(),
                                                          HtsFile::OutputMode::BAM, 0, false));
}

CATCH_TEST_CASE("FileMergeBatcher: Single batch", TEST_GROUP) {
    auto files = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 0);
    utils::FileMergeBatcher batcher(files, filepath("folder", "merged.bam"), 4);
    CATCH_REQUIRE(batcher.num_batches() == 1);
    CATCH_CHECK(batcher.get_recursion_level() == 1);
    CATCH_CHECK(batcher.get_merge_filename(0) == filepath("folder", "merged.bam"));
    const auto& batch = batcher.get_batch(0);
    CATCH_REQUIRE(batch.size() == 4);
    CATCH_CHECK(batch == files);
}

CATCH_TEST_CASE("FileMergeBatcher: Five batches, 2 recursions", TEST_GROUP) {
    auto files = get_dummy_filenames(filepath("folder", "file_"), ".bam", 13, 0);
    utils::FileMergeBatcher batcher(files, filepath("folder", "merged.bam"), 4);
    CATCH_REQUIRE(batcher.num_batches() == 5);
    CATCH_CHECK(batcher.get_recursion_level() == 2);
    CATCH_CHECK(batcher.get_merge_filename(0) == filepath("folder", "batch_0.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(1) == filepath("folder", "batch_1.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(2) == filepath("folder", "batch_2.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(3) == filepath("folder", "batch_3.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(4) == filepath("folder", "merged.bam"));

    // First recursion, 13 files in 2 batches of 4, 1 batch of 3, and 1 batch of 2.
    auto expected_files0 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 0);
    const auto& batch0 = batcher.get_batch(0);
    CATCH_REQUIRE(batch0.size() == 4);
    CATCH_CHECK(batch0 == expected_files0);

    auto expected_files1 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 4);
    const auto& batch1 = batcher.get_batch(1);
    CATCH_REQUIRE(batch1.size() == 4);
    CATCH_CHECK(batch1 == expected_files1);

    auto expected_files2 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 3, 8);
    const auto& batch2 = batcher.get_batch(2);
    CATCH_REQUIRE(batch2.size() == 3);
    CATCH_CHECK(batch2 == expected_files2);

    auto expected_files3 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 2, 11);
    const auto& batch3 = batcher.get_batch(3);
    CATCH_REQUIRE(batch3.size() == 2);
    CATCH_CHECK(batch3 == expected_files3);

    // Second recursion, 1 batch with the 4 merged files from the 1st 4 batches.
    auto expected_files4 = get_dummy_filenames(filepath("folder", "batch_"), ".tmp", 4, 0);
    const auto& batch4 = batcher.get_batch(4);
    CATCH_REQUIRE(batch4.size() == 4);
    CATCH_CHECK(batch4 == expected_files4);
}

CATCH_TEST_CASE("FileMergeBatcher: Eight batches, 3 recursions", TEST_GROUP) {
    auto files = get_dummy_filenames(filepath("folder", "file_"), ".bam", 20, 0);
    utils::FileMergeBatcher batcher(files, filepath("folder", "merged.bam"), 4);
    CATCH_REQUIRE(batcher.num_batches() == 8);
    CATCH_CHECK(batcher.get_recursion_level() == 3);
    CATCH_CHECK(batcher.get_merge_filename(0) == filepath("folder", "batch_0.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(1) == filepath("folder", "batch_1.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(2) == filepath("folder", "batch_2.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(3) == filepath("folder", "batch_3.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(4) == filepath("folder", "batch_4.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(5) == filepath("folder", "batch_5.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(6) == filepath("folder", "batch_6.tmp"));
    CATCH_CHECK(batcher.get_merge_filename(7) == filepath("folder", "merged.bam"));

    // First recursion: 20 files in 5 batches of 4.
    auto expected_files0 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 0);
    const auto& batch0 = batcher.get_batch(0);
    CATCH_REQUIRE(batch0.size() == 4);
    CATCH_CHECK(batch0 == expected_files0);

    auto expected_files1 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 4);
    const auto& batch1 = batcher.get_batch(1);
    CATCH_REQUIRE(batch1.size() == 4);
    CATCH_CHECK(batch1 == expected_files1);

    auto expected_files2 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 8);
    const auto& batch2 = batcher.get_batch(2);
    CATCH_REQUIRE(batch2.size() == 4);
    CATCH_CHECK(batch2 == expected_files2);

    auto expected_files3 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 12);
    const auto& batch3 = batcher.get_batch(3);
    CATCH_REQUIRE(batch3.size() == 4);
    CATCH_CHECK(batch3 == expected_files3);

    auto expected_files4 = get_dummy_filenames(filepath("folder", "file_"), ".bam", 4, 16);
    const auto& batch4 = batcher.get_batch(4);
    CATCH_REQUIRE(batch4.size() == 4);
    CATCH_CHECK(batch4 == expected_files4);

    // Second recursion: 1 batch with 3 files, and 1 batch with 2 files.
    auto expected_files5 = get_dummy_filenames(filepath("folder", "batch_"), ".tmp", 3, 0);
    const auto& batch5 = batcher.get_batch(5);
    CATCH_REQUIRE(batch5.size() == 3);
    CATCH_CHECK(batch5 == expected_files5);

    auto expected_files6 = get_dummy_filenames(filepath("folder", "batch_"), ".tmp", 2, 3);
    const auto& batch6 = batcher.get_batch(6);
    CATCH_REQUIRE(batch6.size() == 2);
    CATCH_CHECK(batch6 == expected_files6);

    // Third recursion: 1 batch with 2 files.
    auto expected_files7 = get_dummy_filenames(filepath("folder", "batch_"), ".tmp", 2, 5);
    const auto& batch7 = batcher.get_batch(7);
    CATCH_REQUIRE(batch7.size() == 2);
    CATCH_CHECK(batch7 == expected_files7);
}

using namespace hts_writer;
using MaybeString = std::optional<std::string>;

namespace {
auto p_cb = utils::ProgressCallback([](float f) { (void)f; });
auto d_cb = utils::DescriptionCallback([](const std::string& s) { (void)s; });
int threads = 0;
const std::string GPU_NAMES = "gpu_names:1";
}  // namespace

CATCH_TEST_CASE(TEST_GROUP "HtsFileWriterBuilder FASTQ happy paths", TEST_GROUP) {
    auto [output_mode, finalise_noop, out_dir] = GENERATE(table<OutputMode, bool, MaybeString>({
            {OutputMode::FASTQ, true, std::nullopt},
            {OutputMode::FASTQ, true, "out"},
    }));

    bool emit_fastq = true;
    bool ref_req = false;
    CATCH_CAPTURE(to_string(output_mode), finalise_noop, emit_fastq, ref_req, out_dir.has_value());

    auto writer_builder = BasecallHtsFileWriterBuilder(emit_fastq, false, false, ref_req, out_dir,
                                                       threads, p_cb, d_cb, GPU_NAMES);
    auto writer = writer_builder.build();

    CATCH_CHECK(writer->get_mode() == output_mode);
    CATCH_CHECK(writer->finalise_is_noop() == finalise_noop);
}

CATCH_TEST_CASE(TEST_GROUP "HtsFileWriterBuilder FASTQ throws given reference", TEST_GROUP) {
    auto [out_dir] = GENERATE(table<MaybeString>({
            {std::nullopt},
            {"out"},
    }));

    bool emit_fastq = true;
    bool ref_req = true;  // << FASTQ cannot store alignment results
    CATCH_CAPTURE(emit_fastq, ref_req, out_dir.has_value());

    CATCH_CHECK_THROWS_AS(BasecallHtsFileWriterBuilder(emit_fastq, false, false, ref_req, out_dir,
                                                       threads, p_cb, d_cb, GPU_NAMES),
                          std::runtime_error);
}

CATCH_TEST_CASE(TEST_GROUP "HtsFileWriterBuilder FASTQ and SAM mutually exclusive", TEST_GROUP) {
    auto [out_dir] = GENERATE(table<MaybeString>({
            {std::nullopt},
            {"out"},
    }));

    bool emit_fastq = true;  // << Both true
    bool emit_sam = true;    // << Both true
    bool ref_req = false;
    CATCH_CAPTURE(emit_fastq, emit_sam, ref_req, out_dir.has_value());

    CATCH_CHECK_THROWS_AS(BasecallHtsFileWriterBuilder(emit_fastq, emit_sam, false, ref_req,
                                                       out_dir, threads, p_cb, d_cb, GPU_NAMES),
                          std::runtime_error);
}

CATCH_TEST_CASE(TEST_GROUP "HtsFileWriterBuilder FASTQ and CRAM mutually exclusive", TEST_GROUP) {
    auto [out_dir] = GENERATE(table<MaybeString>({
            {std::nullopt},
            {"out"},
    }));

    bool emit_fastq = true;  // << Both true
    bool emit_cram = true;   // << Both true
    bool ref_req = false;
    CATCH_CAPTURE(emit_fastq, emit_cram, ref_req, out_dir.has_value());

    CATCH_CHECK_THROWS_AS(BasecallHtsFileWriterBuilder(emit_fastq, false, emit_cram, ref_req,
                                                       out_dir, threads, p_cb, d_cb, GPU_NAMES),
                          std::runtime_error);
}

CATCH_TEST_CASE(TEST_GROUP " HtsFileWriterBuilder SAM happy paths", TEST_GROUP) {
    auto [output_mode, finalise_noop, ref_req, out_dir] =
            GENERATE(table<OutputMode, bool, bool, MaybeString>({
                    // SAM / BAM will sort if writing to file AND reference is set
                    {OutputMode::SAM, true, false, std::nullopt},
                    {OutputMode::SAM, true, false, "out"},
                    {OutputMode::SAM, true, true, std::nullopt},
                    {OutputMode::SAM, false, true, "out"},
            }));

    bool emit_sam = true;
    CATCH_CAPTURE(to_string(output_mode), finalise_noop, emit_sam, ref_req, out_dir.has_value());

    auto writer_builder = BasecallHtsFileWriterBuilder(false, emit_sam, false, ref_req, out_dir,
                                                       threads, p_cb, d_cb, GPU_NAMES);
    auto writer = writer_builder.build();

    CATCH_CHECK(writer->get_mode() == output_mode);
    CATCH_CHECK(writer->finalise_is_noop() == finalise_noop);
}

class HtsFileWriterBuilderTest : public HtsFileWriterBuilder {
public:
    HtsFileWriterBuilderTest(bool emit_fastq,
                             bool emit_sam,
                             bool emit_cram,
                             bool reference_requested,
                             const std::optional<std::string>& output_dir,
                             int writer_threads,
                             utils::ProgressCallback progress_callback,
                             utils::DescriptionCallback description_callback,
                             std::string gpu_names,
                             bool is_fd_tty,
                             bool is_fd_pipe)
            : HtsFileWriterBuilder(emit_fastq,
                                   emit_sam,
                                   emit_cram,
                                   reference_requested,
                                   output_dir,
                                   writer_threads,
                                   std::move(progress_callback),
                                   std::move(description_callback),
                                   std::move(gpu_names),
                                   false) {
        m_is_fd_tty = is_fd_tty;
        m_is_fd_pipe = is_fd_pipe;
    };
};

CATCH_TEST_CASE(TEST_GROUP " HtsFileWriterBuilder tty and pipe settings", TEST_GROUP) {
    auto [output_mode, ref_req, out_dir, emit_sam, is_fd_tty, is_fd_pipe] =
            GENERATE(table<OutputMode, bool, MaybeString, bool, bool, bool>({
                    // Always write SAM if writing to stdout tty
                    {OutputMode::SAM, false, std::nullopt, true, true, false},
                    {OutputMode::SAM, true, std::nullopt, true, true, false},
                    {OutputMode::SAM, false, std::nullopt, false, true, false},
                    {OutputMode::SAM, true, std::nullopt, false, true, false},

                    // Output depends on --emit-sam regardless of reference when piping
                    {OutputMode::SAM, false, std::nullopt, true, false, true},
                    {OutputMode::SAM, true, std::nullopt, true, false, true},
                    {OutputMode::UBAM, false, std::nullopt, false, false, true},
                    {OutputMode::UBAM, true, std::nullopt, false, false, true},

            }));

    CATCH_CAPTURE(to_string(output_mode), emit_sam, ref_req, out_dir.has_value(), is_fd_tty,
                  is_fd_pipe);

    auto writer_builder =
            HtsFileWriterBuilderTest(false, emit_sam, false, ref_req, out_dir, threads, p_cb, d_cb,
                                     GPU_NAMES, is_fd_tty, is_fd_pipe);
    auto writer = writer_builder.build();

    CATCH_CHECK(is_fd_tty != is_fd_pipe);
    CATCH_CHECK(writer->get_mode() == output_mode);
}

CATCH_TEST_CASE(TEST_GROUP " HtsFileWriterBuilder BAM/CRAM happy paths", TEST_GROUP) {
    auto [finalise_noop, ref_req, out_dir] = GENERATE(table<bool, bool, MaybeString>({
            {true, false, std::nullopt},
            {true, false, "out"},
            {true, true, std::nullopt},
            // BAM/CRAM will sort if writing to file AND reference is set
            {false, true, "out"},
    }));

    auto [output_mode] = GENERATE(table<OutputMode>({OutputMode::BAM, OutputMode::CRAM}));

    bool is_fd_tty = false;
    bool is_fd_pipe = false;
    CATCH_CAPTURE(to_string(output_mode), finalise_noop, ref_req, out_dir.has_value(), is_fd_tty,
                  is_fd_pipe);

    auto writer_builder = HtsFileWriterBuilderTest(false, false, output_mode == OutputMode::CRAM,
                                                   ref_req, out_dir, threads, p_cb, d_cb, GPU_NAMES,
                                                   is_fd_tty, is_fd_pipe);

    auto writer = writer_builder.build();

    CATCH_CHECK(writer->get_mode() == output_mode);
    CATCH_CHECK(writer->finalise_is_noop() == finalise_noop);
}

CATCH_TEST_CASE(TEST_GROUP " HtsFileWriter getters ", TEST_GROUP) {
    int writer_threads = 10;
    size_t progress_res = 0;
    auto progress_cb =
            utils::ProgressCallback([&progress_res](float value) { progress_res = value; });

    std::string description_res{};
    auto description_cb = utils::DescriptionCallback(
            [&description_res](const std::string& value) { description_res = value; });
    auto writer_builder =
            BasecallHtsFileWriterBuilder(true, false, false, false, std::nullopt, writer_threads,
                                         progress_cb, description_cb, GPU_NAMES);
    auto writer = writer_builder.build();

    auto& writer_ref = *writer;
    CATCH_CHECK(typeid(writer_ref) == typeid(StreamHtsFileWriter));

    CATCH_CHECK(writer->get_gpu_names() == "gpu:" + GPU_NAMES);

    const int test_progress = 100;
    writer->set_progress(test_progress);
    CATCH_CHECK(progress_res == test_progress);

    const auto test_description = "running a test";
    writer->set_description(test_description);
    CATCH_CHECK(description_res == test_description);
}

CATCH_TEST_CASE(TEST_GROUP " Writer Structures and Strategies", TEST_GROUP) {
    using namespace hts_writer;
    namespace fs = std::filesystem;

    const auto tmp_dir = make_temp_dir("test_writer_structure");
    const auto root = tmp_dir.m_path.string();

    auto [output_mode] = GENERATE(table<OutputMode>({
            {OutputMode::FASTQ},
            {OutputMode::SAM},
            {OutputMode::BAM},
            {OutputMode::CRAM},
            {OutputMode::UBAM},
    }));

    hts_writer::SingleFileStructure structure(root, output_mode);
    const auto& single_path = structure.get_path(HtsData{});

    CATCH_CAPTURE(root, single_path, output_mode);

    // Root is the parent of the output
    const bool is_parent = fs::path(single_path).parent_path() == fs::path(root);
    CATCH_CHECK(is_parent);

    // Output starts with 'calls_'
    const auto fname = fs::path(single_path).filename().string();
    const bool fname_has_calls_prefix = fname.find("calls_") != std::string::npos;
    CATCH_CAPTURE(fname);
    CATCH_CHECK(fname_has_calls_prefix);

    // Output file extension matches the output mode
    const auto extension = fs::path(single_path).filename().extension().string();
    CATCH_CAPTURE(extension, get_suffix(output_mode));
    CATCH_CHECK(extension == get_suffix(output_mode));

    // Regardless of the input data the output path should be the same for SingleFileStructure
    const auto& single_path2 =
            structure.get_path(HtsData{nullptr, {"kit", "", "", "", "FLOWCELL"}, nullptr});
    CATCH_CHECK(single_path == single_path2);
}

CATCH_TEST_CASE(TEST_GROUP " Writer Nested Structures No Barcodes", TEST_GROUP) {
    using namespace hts_writer;
    namespace fs = std::filesystem;

    const auto tmp_dir = make_temp_dir("test_writer_nested_structure");
    const auto root = tmp_dir.m_path.string();

    // subfolder here is NOT the file extension but the output folder name
    auto [output_mode, subfolder] = GENERATE(table<OutputMode, std::string>({
            {OutputMode::FASTQ, "fastq"},
            {OutputMode::SAM, "bam"},
            {OutputMode::BAM, "bam"},
            {OutputMode::UBAM, "bam"},
            {OutputMode::CRAM, "bam"},
    }));

    auto [attrs, expected_dir, expected_stem] = GENERATE_COPY(table<HtsData::ReadAttributes,
                                                                    std::string, std::string>(

            {{
                     HtsData::ReadAttributes{
                             "sequencing_kit",
                             "experiment-id",
                             "sample-id",
                             "position-id",
                             "flowcell-id",
                             "protocol-id",
                             "acquisition-id",
                             "",
                             "",
                             0,
                             0,
                     },
                     "experiment-id/sample-id/19700101_0000_position-id_flowcell-id_protocol/" +
                             subfolder + "_pass",
                     "flowcell-id_pass_protocol_acquisit_0",
             },
             {
                     HtsData::ReadAttributes{
                             "kit",
                             "exp",
                             "sample",
                             "pos",
                             "fc",
                             "proto",
                             "acq",
                             "",
                             "",
                             946782245000 /* 2000/01/02 03:04:05 */,
                             1,
                     },
                     "exp/sample/20000102_0304_pos_fc_proto/" + subfolder + "_pass",
                     "fc_pass_proto_acq_0",
             }}));

    hts_writer::NestedFileStructure structure(root, output_mode, false);
    const auto path = fs::path(structure.get_path(HtsData{nullptr, attrs}));
    CATCH_CAPTURE(root, path, output_mode, attrs.experiment_id);

    // Check root is the parent of the output
    CATCH_CHECK(path.string().substr(0, root.size()) == fs::path(root).string());
    // Check the folder structure
    CATCH_CHECK(fs::relative(path.parent_path(), root) == fs::path(expected_dir));
    // Check the filename
    CATCH_CHECK(path.stem() == expected_stem);
    // Check the file type / extension
    const std::string extension = get_suffix(output_mode);
    CATCH_CHECK(path.extension() == extension);
}

CATCH_TEST_CASE(TEST_GROUP " Writer Nested Structures with Barcodes", TEST_GROUP) {
    using namespace hts_writer;
    namespace fs = std::filesystem;

    const auto tmp_dir = make_temp_dir("test_writer_nested_structure");
    const auto root = tmp_dir.m_path.string();

    auto [output_mode, subfolder] = GENERATE(table<OutputMode, std::string>({
            {OutputMode::FASTQ, "fastq"},
            {OutputMode::BAM, "bam"},
            {OutputMode::SAM, "bam"},
            {OutputMode::CRAM, "bam"},
    }));

    auto [barcode_name, alias, description] =
            GENERATE_COPY(table<std::string, std::string, std::string>({
                    {"", "", "test without barcode or alias"},
                    {"barcode99", "", "test with barcode but without alias"},
                    {"unclassified", "", "test unclassified read"},
                    {"barcode01", "patient_id_5", "test with barcode and alias"},
            }));

    auto barcode_score_result = std::make_shared<BarcodeScoreResult>();
    barcode_score_result->barcode_name = barcode_name;
    if (alias.empty()) {
        alias = barcode_name;
    } else {
        barcode_score_result->alias = alias;
    }

    const HtsData::ReadAttributes attrs{
            "SQK-RBK114-96",
            "",
            "barcoding_run",
            "fc-pos",
            "PAO25751",
            "proto-id",
            "acq-id",
            "",
            "",
            946782245000 /* 2000/01/02 03:04:05 */,
            0,
    };

    const std::string expected_base =
            "barcoding_run/20000102_0304_fc-pos_PAO25751_proto-id/" + subfolder + "_pass";

    const auto expected_classification_folder =
            alias.empty() ? fs::path("unclassified") : fs::path(alias);

    std::ostringstream oss;
    oss << "PAO25751" << "_pass_" << alias << (alias.empty() ? "unclassified_" : "_")
        << "proto-id_acq-id_0";
    const std::string expected_fname = oss.str();

    hts_writer::NestedFileStructure structure(root, output_mode, true);
    const auto path = fs::path(structure.get_path(HtsData{nullptr, attrs, barcode_score_result}));

    CATCH_SECTION(description) {
        CATCH_CAPTURE(root, path, output_mode, barcode_name);
        // Check root is the parent of the output
        CATCH_CHECK(path.string().substr(0, root.size()) == root);
        // Check the filename
        CATCH_CHECK(path.stem() == expected_fname);
        // Check the file type / extension
        const std::string extension = get_suffix(output_mode);
        CATCH_CHECK(path.extension() == extension);

        // Expect an additional subdir for the classification
        const auto base = path.parent_path().parent_path();
        // Check the folder base structure (excluding the classification)
        const auto relative_base_folder = fs::relative(base, root);
        CATCH_CHECK(relative_base_folder == fs::path(expected_base));

        // Check the classification subdir should be the alias name
        const auto classification_folder = fs::relative(path, base).parent_path();
        CATCH_CHECK(classification_folder == expected_classification_folder);
    }
}

CATCH_TEST_CASE(TEST_GROUP " Writer Nested Structures with Barcodes from file", TEST_GROUP) {
    using namespace hts_writer;
    namespace fs = std::filesystem;

    const auto tmp_dir = make_temp_dir("test_writer_nested_structure");
    const auto root = tmp_dir.m_path.string();

    auto [output_mode, subfolder] = GENERATE(table<OutputMode, std::string>({
            {OutputMode::FASTQ, "fastq"},
            {OutputMode::BAM, "bam"},
            {OutputMode::SAM, "bam"},
            {OutputMode::CRAM, "bam"},
    }));

    auto [barcode_name, alias, description] =
            GENERATE_COPY(table<std::string, std::string, std::string>({
                    {"", "", "test without barcode or alias"},
                    {"barcode99", "barcode99", "test with barcode but without alias"},
                    {"unclassified", "unclassified", "test unclassified read"},
                    {"barcode01", "patient_id_5", "test with barcode and alias"},
            }));
    auto add_kit_name = GENERATE(false, true);

    const HtsData::ReadAttributes attrs{
            "SQK-RBK114-96",
            "",
            "barcoding_run",
            "fc-pos",
            "PAO25751",
            "proto-id",
            "acq-id",
            "",
            "",
            946782245000 /* 2000/01/02 03:04:05 */,
            0,
    };

    const std::string expected_base =
            "barcoding_run/20000102_0304_fc-pos_PAO25751_proto-id/" + subfolder + "_pass";

    const auto expected_classification_folder =
            alias.empty() ? fs::path("unclassified") : fs::path(alias);

    std::ostringstream oss;
    oss << "PAO25751" << "_pass_" << alias << (alias.empty() ? "unclassified_" : "_")
        << "proto-id_acq-id_0";
    const std::string expected_fname = oss.str();

    hts_writer::NestedFileStructure structure(root, output_mode, true);
    BamPtr bam_record(bam_init1());
    const std::string qname = "test_record";
    // fill read record with minimal information
    bam_set1(bam_record.get(), qname.length(), qname.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0,
             1, "*", "*", 0);

    if (!alias.empty() && alias != "unclassified") {
        const auto barcode = add_kit_name ? "SQK-RBK114-96_" + alias : alias;
        bam_aux_append(bam_record.get(), "BC", 'Z', barcode.length() + 1,
                       reinterpret_cast<const uint8_t*>(barcode.c_str()));
    }
    const auto path = fs::path(structure.get_path(HtsData{std::move(bam_record), attrs, nullptr}));

    CATCH_SECTION(description) {
        CATCH_CAPTURE(root, path, output_mode, barcode_name, add_kit_name);
        // Check root is the parent of the output
        CATCH_CHECK(path.string().substr(0, root.size()) == root);
        // Check the filename
        CATCH_CHECK(path.stem() == expected_fname);
        // Check the file type / extension
        const std::string extension = get_suffix(output_mode);
        CATCH_CHECK(path.extension() == extension);

        // Expect an additional subdir for the classification or from unclassified
        const auto base = path.parent_path().parent_path();
        // Check the folder base structure (excluding the classification)
        const auto relative_base_folder = fs::relative(base, root);
        CATCH_CHECK(relative_base_folder == fs::path(expected_base));

        // Check the classification subdir should be the alias name
        const auto classification_folder = fs::relative(path, base).parent_path();
        CATCH_CHECK(classification_folder == expected_classification_folder);
    }
}

CATCH_TEST_CASE("HtsReaderTest: parse htslib header line type", TEST_GROUP) {
    using HLT = utils::HeaderLineType;
    auto [header_line, expected] = GENERATE(table<std::string, HLT>({
            {"@CO a very interesting comment indeed", HLT::CO},
            {"@HD key=value", HLT::HD},
            {"@PG?! CL:dorado basecaller", HLT::PG},
            {"@RG CO HD PG SQ", HLT::RG},
            {"@SQ\tSN:reference\tLN:2000", HLT::SQ},
            {"RG", HLT::UNKNOWN},
            {"CO @CO", HLT::UNKNOWN},
            {"", HLT::UNKNOWN},

    }));
    CATCH_CAPTURE(header_line);
    const auto result = dorado::utils::parse_header_line_type(header_line);
    CATCH_CHECK(result == expected);
}

CATCH_TEST_CASE("HtsReaderTest: parse htslib header", TEST_GROUP) {
    using HLT = utils::HeaderLineType;
    const std::string example_1 =
            "@HD\tVN:1.6\tSO:coordinate\n"
            "@PG\tID:aligner\tPN:minimap2\tVN:2.26-r1175zn\n"
            "@RG\tID:run1_model1\tDT:2022-10-20T14:48:32Z\tDS:runid=run1 "
            "basecall_model=model1\tLB:NA12878\tPL:ONT\tPU:SOMEBODY\tal:NA12878\n"
            "@SQ\tSN:ref1\tLN:1000\n"
            "@CO\tfoo\n"
            "@CO\tbar\n";

    CATCH_SECTION("parse all") {
        const auto expected_types =
                std::vector({HLT::HD, HLT::PG, HLT::RG, HLT::SQ, HLT::CO, HLT::CO});
        const auto lines = utils::parse_header(example_1.c_str(), {});
        CATCH_CHECK(lines.size() == expected_types.size());
        for (size_t i = 0; i < expected_types.size(); i++) {
            CATCH_CHECK(expected_types[i] == lines[i].header_type);
        }
    }

    CATCH_SECTION("parse selection") {
        const auto selection = std::set({HLT::SQ, HLT::RG});
        const auto expected_types = std::vector({HLT::RG, HLT::SQ});
        const auto lines = utils::parse_header(example_1.c_str(), selection);
        CATCH_CHECK(lines.size() == expected_types.size());
        for (size_t i = 0; i < expected_types.size(); i++) {
            CATCH_CHECK(expected_types[i] == lines[i].header_type);
        }
    }
}
