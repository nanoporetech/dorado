#include "alignment/Minimap2Index.h"

#include "TestUtils.h"
#include "alignment/minimap2_args.h"
#include "alignment/minimap2_wrappers.h"
#include "read_pipeline/HtsWriter.h"
#include "utils/hts_file.h"
#include "utils/stream_utils.h"
#include "utils/types.h"

#include <htslib/sam.h>

#include <filesystem>

// Catch must come last so we can undo torch defining CHECK.
#undef CHECK
#include <catch2/catch_all.hpp>

#define TEST_GROUP "[alignment::Minimap2Index]"

using namespace dorado::utils;

namespace {

class Minimap2IndexTestFixture {
protected:
    dorado::alignment::Minimap2Index cut{};
    std::string reference_file;
    std::string empty_file;

public:
    Minimap2IndexTestFixture() {
        const std::string read_id{"aligner_node_test"};
        std::filesystem::path aligner_test_dir{get_aligner_data_dir()};
        auto ref = aligner_test_dir / "target.fq";
        reference_file = ref.string();
        auto empty = aligner_test_dir / "empty.fa";
        empty_file = empty.string();

        cut.initialise(dorado::alignment::create_dflt_options());
    }
};

}  // namespace

namespace dorado::alignment::test {

TEST_CASE(TEST_GROUP " initialise() with default options does not throw", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE_NOTHROW(cut.initialise(create_dflt_options()));
}

TEST_CASE(TEST_GROUP " initialise() with default options returns true", TEST_GROUP) {
    Minimap2Index cut{};

    REQUIRE(cut.initialise(create_dflt_options()));
}

TEST_CASE(TEST_GROUP " initialise() with specified option sets indexing options", TEST_GROUP) {
    Minimap2Index cut{};

    auto options{create_dflt_options()};
    options.index_options->get().k = 11;
    options.index_options->get().w = 12;
    options.index_options->get().batch_size = 123456789;

    cut.initialise(options);

    CHECK(cut.index_options().k == options.index_options->get().k);
    CHECK(cut.index_options().w == options.index_options->get().w);
    CHECK(cut.index_options().batch_size == options.index_options->get().batch_size);
}

TEST_CASE(TEST_GROUP " initialise() with default options sets mapping options", TEST_GROUP) {
    Minimap2Index cut{};
    auto options{create_dflt_options()};
    options.mapping_options->get().bw = 300;
    options.mapping_options->get().bw_long = 12000;

    cut.initialise(options);

    CHECK(cut.mapping_options().bw == options.mapping_options->get().bw);
    CHECK(cut.mapping_options().bw_long == options.mapping_options->get().bw_long);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " load() with invalid reference file returns reference_file_not_found",
                 TEST_GROUP) {
    REQUIRE(cut.load("some_reference_file", 1, false) == IndexLoadResult::reference_file_not_found);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " load() with empty reference file returns end_of_index",
                 TEST_GROUP) {
    REQUIRE(cut.load(empty_file, 1, false) == IndexLoadResult::end_of_index);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " load() with valid reference file returns success",
                 TEST_GROUP) {
    REQUIRE(cut.load(reference_file, 1, false) == IndexLoadResult::success);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP
                 " get_options() after successful load() compares as equal to default options",
                 TEST_GROUP) {
    CHECK(cut.load(reference_file, 1, false) == IndexLoadResult::success);

    REQUIRE(cut.get_options() == dorado::alignment::create_dflt_options());
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP " create_compatible_index() with valid options returns non-null",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{create_dflt_options()};
    compatible_options.mapping_options->get().best_n = cut.mapping_options().best_n + 1;

    REQUIRE(cut.create_compatible_index(compatible_options) != nullptr);
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP
                 " create_compatible_index() with valid options returns Minimap2Index with same "
                 "underlying index",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{create_dflt_options()};
    compatible_options.mapping_options->get().best_n = cut.mapping_options().best_n + 1;

    auto compatible_index = cut.create_compatible_index(compatible_options);

    REQUIRE(compatible_index->index() == cut.index());
}

TEST_CASE_METHOD(Minimap2IndexTestFixture,
                 TEST_GROUP
                 " create_compatible_index() with valid options returns Minimap2Index with mapping "
                 "options updated",
                 TEST_GROUP) {
    cut.load(reference_file, 1, false);
    Minimap2Options compatible_options{create_dflt_options()};
    compatible_options.mapping_options->get().best_n = cut.mapping_options().best_n + 1;

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

    Minimap2Index cut{};
    cut.initialise(mm2::parse_options("-I 10k"));

    SECTION("No split index allowed") {
        CHECK(cut.load(temp_input_file.string(), 1, false) ==
              IndexLoadResult::split_index_not_supported);
    }

    SECTION("Split index allowed") {
        CHECK(cut.load(temp_input_file.string(), 1, true) == IndexLoadResult::success);
        CHECK(cut.load_next_chunk(1) == IndexLoadResult::success);
    }
}

}  // namespace dorado::alignment::test
