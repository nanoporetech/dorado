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
