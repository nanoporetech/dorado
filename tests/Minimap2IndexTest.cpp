#include "alignment/Minimap2Index.h"

#include "TestUtils.h"
#include "read_pipeline/HtsWriter.h"
#include "utils/hts_file.h"
#include "utils/stream_utils.h"
#include "utils/types.h"

#include <catch2/catch.hpp>
#include <htslib/sam.h>

#include <filesystem>

#define TEST_GROUP "[alignment::Minimap2Index]"

using namespace dorado::utils;

namespace {

class Minimap2IndexTestFixture {
protected:
    dorado::alignment::Minimap2Index cut{};
    std::string reference_file;

public:
    Minimap2IndexTestFixture() {
        const std::string read_id{"aligner_node_test"};
        std::filesystem::path aligner_test_dir{get_aligner_data_dir()};
        auto ref = aligner_test_dir / "target.fq";
        reference_file = ref.string();

        cut.initialise(dorado::alignment::dflt_options);
    }
};

}  // namespace

namespace dorado::alignment::test {

TEST_CASE(TEST_GROUP " initialise() with default options does not throw", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE_NOTHROW(cut.initialise(dflt_options));
}

TEST_CASE(TEST_GROUP " initialise() with default options returns true", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE(cut.initialise(dflt_options));
}

TEST_CASE(TEST_GROUP " initialise() with specified option sets indexing options", TEST_GROUP) {
    Minimap2Index cut{};

    auto options{dflt_options};
    options.kmer_size = 11;
    options.window_size = 12;
    options.index_batch_size = 123456789;

    cut.initialise(options);

    CHECK(cut.index_options().k == *options.kmer_size);
    CHECK(cut.index_options().w == *options.window_size);
    CHECK(cut.index_options().batch_size == *options.index_batch_size);
}

TEST_CASE(TEST_GROUP " initialise() with default options sets mapping options", TEST_GROUP) {
    Minimap2Index cut{};
    auto options{dflt_options};
    options.bandwidth = 300;
    options.bandwidth_long = 12000;
    options.best_n_secondary = 123456789;
    options.soft_clipping = true;

    cut.initialise(options);

    CHECK(cut.mapping_options().bw == options.bandwidth);
    CHECK(cut.mapping_options().bw_long == options.bandwidth_long);
    CHECK(cut.mapping_options().best_n == options.best_n_secondary);
    CHECK(cut.mapping_options().flag > 0);  // Just checking it's been updated.
}

TEST_CASE(TEST_GROUP " initialise() with invalid options returns false", TEST_GROUP) {
    Minimap2Index cut{};
    auto invalid_options{dflt_options};
    invalid_options.bandwidth_long = 100;
    invalid_options.bandwidth = *invalid_options.bandwidth_long + 1;

    bool result{};
    SuppressStderr::invoke(
            [&result, &invalid_options, &cut] { result = cut.initialise(invalid_options); });

    REQUIRE_FALSE(result);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " load() with invalid reference file returns reference_file_not_found",
                 TEST_GROUP) {
    REQUIRE(cut.load("some_reference_file", 1, false) == IndexLoadResult::reference_file_not_found);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " load() with valid reference file returns success",
                 TEST_GROUP) {
    REQUIRE(cut.load(reference_file, 1, false) == IndexLoadResult::success);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " create_compatible_index() with invalid mapping options returns null",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options invalid_compatible_options{dflt_options};
    invalid_compatible_options.bandwidth_long = 100;
    invalid_compatible_options.bandwidth = *invalid_compatible_options.bandwidth_long + 1;

    std::shared_ptr<Minimap2Index> compatible_index{};
    {
        SuppressStderr suppressed{};
        compatible_index = cut.create_compatible_index(invalid_compatible_options);
    }

    REQUIRE(compatible_index == nullptr);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " create_compatible_index() with valid options returns non-null",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{dflt_options};
    compatible_options.best_n_secondary = cut.mapping_options().best_n + 1;

    REQUIRE(cut.create_compatible_index(compatible_options) != nullptr);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP
                 " create_compatible_index() with valid options returns Minimap2Index with same "
                 "underlying index",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{dflt_options};
    compatible_options.best_n_secondary = cut.mapping_options().best_n + 1;

    auto compatible_index = cut.create_compatible_index(compatible_options);

    REQUIRE(compatible_index->index() == cut.index());
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP
                 " create_compatible_index() with valid options returns Minimap2Index with mapping "
                 "options updated",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{dflt_options};
    compatible_options.best_n_secondary = cut.mapping_options().best_n + 1;

    auto compatible_index = cut.create_compatible_index(compatible_options);

    REQUIRE(compatible_index->mapping_options().best_n == cut.mapping_options().best_n + 1);
}

TEST_CASE(TEST_GROUP " Test split index loading", TEST_GROUP) {
    // Create large index file
    auto temp_dir = tests::make_temp_dir("mm2_split_index_test");
    auto temp_input_file = temp_dir.m_path / "input.fa";
    utils::HtsFile hts_file(temp_input_file.string(), utils::HtsFile::OutputMode::FASTA, 2, false);
    HtsWriter writer(hts_file, "");
    for (auto& [seq, read_id] : {
                 std::make_pair<std::string, std::string>(generate_random_sequence_string(10000),
                                                          "read1"),
                 std::make_pair<std::string, std::string>(generate_random_sequence_string(10000),
                                                          "read2"),
                 std::make_pair<std::string, std::string>(generate_random_sequence_string(10000),
                                                          "read3"),
                 std::make_pair<std::string, std::string>(generate_random_sequence_string(10000),
                                                          "read4"),
                 std::make_pair<std::string, std::string>(generate_random_sequence_string(10000),
                                                          "read5"),
         }) {
        BamPtr rec = BamPtr(bam_init1());
        bam_set1(rec.get(), read_id.length(), read_id.c_str(), 4, -1, -1, 0, 0, nullptr, -1, -1, 0,
                 seq.length(), seq.c_str(), nullptr, 0);
        writer.write(rec.get());
    }
    hts_file.finalise([](size_t) { /* noop */ });

    SECTION("No split index allowed") {
        Minimap2Index cut{};

        auto options{dflt_options};
        options.index_batch_size = 10000;

        cut.initialise(options);

        CHECK(cut.load(temp_input_file.string(), 1, false) ==
              IndexLoadResult::split_index_not_supported);
    }

    SECTION("Split index allowed") {
        Minimap2Index cut{};

        auto options{dflt_options};
        options.index_batch_size = 10000;

        cut.initialise(options);

        CHECK(cut.load(temp_input_file.string(), 1, true) == IndexLoadResult::success);
        CHECK(cut.load_next_chunk(1) == IndexLoadResult::success);
    }
}

}  // namespace dorado::alignment::test
