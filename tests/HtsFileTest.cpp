#include "TestUtils.h"
#include "utils/PostCondition.h"
#include "utils/hts_file.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
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
            REQUIRE(index < records.size());
            if (is_sorted) {
                auto sorting_key = HtsFile::calculate_sorting_key(record.get());
                REQUIRE(sorting_key >= last_sorting_key);
                last_sorting_key = sorting_key;
            } else {
                // Output records should be in the order they were written.
                auto expected_record = records[indices[index]].get();
                std::string qname(bam_get_qname(record.get()));
                std::string expected_qname(bam_get_qname(expected_record));
                REQUIRE(qname == expected_qname);
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
        files[i] = base_name + std::to_string(i + first_index) + ext;
    }
    return files;
}

}  // namespace

TEST_CASE("HtsFileTest: Write to unsorted file", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    int callback_calls = tester.write_output_records(0);
    REQUIRE(callback_calls == 2);

    tester.check_output(false);
}

TEST_CASE("HtsFileTest: Write to single sorted file", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    // A 5 MB buffer should make sure only a single temp file is written.
    int callback_calls = tester.write_output_records(5000000);
    REQUIRE(callback_calls == 3);

    tester.check_output(true);
}

TEST_CASE("HtsFileTest: Write to multiple sorted files, and merge", TEST_GROUP) {
    Tester tester;
    tester.read_input_records();

    // A 200 KB buffer should make sure multiple temp files are written.
    int callback_calls = tester.write_output_records(200000);
    REQUIRE(callback_calls > 4);

    tester.check_output(true);
}

TEST_CASE("HtsFileTest: construct with zero threads for sorted BAM does not throw", TEST_GROUP) {
    Tester tester;
    std::unique_ptr<HtsFile> cut{};
    auto finalize_file = utils::PostCondition([&cut] { cut->finalise([](size_t) {}); });

    REQUIRE_NOTHROW(cut = std::make_unique<HtsFile>(tester.file_out_path.string(),
                                                    HtsFile::OutputMode::BAM, 0, true));
}

TEST_CASE("HtsFileTest: construct with zero threads for unsorted BAM does not throw", TEST_GROUP) {
    Tester tester;
    std::unique_ptr<HtsFile> cut{};
    auto finalize_file = utils::PostCondition([&cut] { cut->finalise([](size_t) {}); });

    REQUIRE_NOTHROW(cut = std::make_unique<HtsFile>(tester.file_out_path.string(),
                                                    HtsFile::OutputMode::BAM, 0, false));
}

TEST_CASE(
        "HtsFileTest: set_num_threads with 2 after constructed with zero threads for unsorted BAM "
        "does not throw",
        TEST_GROUP) {
    Tester tester;
    std::unique_ptr<HtsFile> cut{};
    auto finalize_file = utils::PostCondition([&cut] { cut->finalise([](size_t) {}); });
    cut = std::make_unique<HtsFile>(tester.file_out_path.string(), HtsFile::OutputMode::BAM, 0,
                                    false);

    cut->set_num_threads(2);
}

TEST_CASE("FileMergeBatcher: Single batch", TEST_GROUP) {
    auto files = get_dummy_filenames("folder/file_", ".bam", 4, 0);
    utils::FileMergeBatcher batcher(files, "folder/merged.bam", 4);
    REQUIRE(batcher.num_batches() == 1);
    CHECK(batcher.get_recursion_level() == 1);
    CHECK(batcher.get_merge_filename(0) == std::string("folder/merged.bam"));
    const auto& batch = batcher.get_batch(0);
    REQUIRE(batch.size() == 4);
    CHECK(batch == files);
}

TEST_CASE("FileMergeBatcher: Five batches, 2 recursions", TEST_GROUP) {
    auto files = get_dummy_filenames("folder/file_", ".bam", 13, 0);
    utils::FileMergeBatcher batcher(files, "folder/merged.bam", 4);
    REQUIRE(batcher.num_batches() == 5);
    CHECK(batcher.get_recursion_level() == 2);
    CHECK(batcher.get_merge_filename(0) == std::string("folder/batch_0.tmp"));
    CHECK(batcher.get_merge_filename(1) == std::string("folder/batch_1.tmp"));
    CHECK(batcher.get_merge_filename(2) == std::string("folder/batch_2.tmp"));
    CHECK(batcher.get_merge_filename(3) == std::string("folder/batch_3.tmp"));
    CHECK(batcher.get_merge_filename(4) == std::string("folder/merged.bam"));

    // First recursion, 13 files in 2 batches of 4, 1 batch of 3, and 1 batch of 2.
    auto expected_files0 = get_dummy_filenames("folder/file_", ".bam", 4, 0);
    const auto& batch0 = batcher.get_batch(0);
    REQUIRE(batch0.size() == 4);
    CHECK(batch0 == expected_files0);

    auto expected_files1 = get_dummy_filenames("folder/file_", ".bam", 4, 4);
    const auto& batch1 = batcher.get_batch(1);
    REQUIRE(batch1.size() == 4);
    CHECK(batch1 == expected_files1);

    auto expected_files2 = get_dummy_filenames("folder/file_", ".bam", 3, 8);
    const auto& batch2 = batcher.get_batch(2);
    REQUIRE(batch2.size() == 3);
    CHECK(batch2 == expected_files2);

    auto expected_files3 = get_dummy_filenames("folder/file_", ".bam", 2, 11);
    const auto& batch3 = batcher.get_batch(3);
    REQUIRE(batch3.size() == 2);
    CHECK(batch3 == expected_files3);

    // Second recursion, 1 batch with the 4 merged files from the 1st 4 batches.
    auto expected_files4 = get_dummy_filenames("folder/batch_", ".tmp", 4, 0);
    const auto& batch4 = batcher.get_batch(4);
    REQUIRE(batch4.size() == 4);
    CHECK(batch4 == expected_files4);
}

TEST_CASE("FileMergeBatcher: Eight batches, 3 recursions", TEST_GROUP) {
    auto files = get_dummy_filenames("folder/file_", ".bam", 20, 0);
    utils::FileMergeBatcher batcher(files, "folder/merged.bam", 4);
    REQUIRE(batcher.num_batches() == 8);
    CHECK(batcher.get_recursion_level() == 3);
    CHECK(batcher.get_merge_filename(0) == std::string("folder/batch_0.tmp"));
    CHECK(batcher.get_merge_filename(1) == std::string("folder/batch_1.tmp"));
    CHECK(batcher.get_merge_filename(2) == std::string("folder/batch_2.tmp"));
    CHECK(batcher.get_merge_filename(3) == std::string("folder/batch_3.tmp"));
    CHECK(batcher.get_merge_filename(4) == std::string("folder/batch_4.tmp"));
    CHECK(batcher.get_merge_filename(5) == std::string("folder/batch_5.tmp"));
    CHECK(batcher.get_merge_filename(6) == std::string("folder/batch_6.tmp"));
    CHECK(batcher.get_merge_filename(7) == std::string("folder/merged.bam"));

    // First recursion: 20 files in 5 batches of 4.
    auto expected_files0 = get_dummy_filenames("folder/file_", ".bam", 4, 0);
    const auto& batch0 = batcher.get_batch(0);
    REQUIRE(batch0.size() == 4);
    CHECK(batch0 == expected_files0);

    auto expected_files1 = get_dummy_filenames("folder/file_", ".bam", 4, 4);
    const auto& batch1 = batcher.get_batch(1);
    REQUIRE(batch1.size() == 4);
    CHECK(batch1 == expected_files1);

    auto expected_files2 = get_dummy_filenames("folder/file_", ".bam", 4, 8);
    const auto& batch2 = batcher.get_batch(2);
    REQUIRE(batch2.size() == 4);
    CHECK(batch2 == expected_files2);

    auto expected_files3 = get_dummy_filenames("folder/file_", ".bam", 4, 12);
    const auto& batch3 = batcher.get_batch(3);
    REQUIRE(batch3.size() == 4);
    CHECK(batch3 == expected_files3);

    auto expected_files4 = get_dummy_filenames("folder/file_", ".bam", 4, 16);
    const auto& batch4 = batcher.get_batch(4);
    REQUIRE(batch4.size() == 4);
    CHECK(batch4 == expected_files4);

    // Second recursion: 1 batch with 3 files, and 1 batch with 2 files.
    auto expected_files5 = get_dummy_filenames("folder/batch_", ".tmp", 3, 0);
    const auto& batch5 = batcher.get_batch(5);
    REQUIRE(batch5.size() == 3);
    CHECK(batch5 == expected_files5);

    auto expected_files6 = get_dummy_filenames("folder/batch_", ".tmp", 2, 3);
    const auto& batch6 = batcher.get_batch(6);
    REQUIRE(batch6.size() == 2);
    CHECK(batch6 == expected_files6);

    // Third recursion: 1 batch with 2 files.
    auto expected_files7 = get_dummy_filenames("folder/batch_", ".tmp", 2, 5);
    const auto& batch7 = batcher.get_batch(7);
    REQUIRE(batch7.size() == 2);
    CHECK(batch7 == expected_files7);
}
