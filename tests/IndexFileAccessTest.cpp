#include "alignment/IndexFileAccess.h"

#include "TestUtils.h"

#include <catch2/catch.hpp>

#include <filesystem>

#define TEST_GROUP "[alignment::IndexFileAccess]"

namespace {

const std::string& valid_reference_file() {
    static const std::string reference_file = []() {
        const std::string read_id{"aligner_node_test"};
        std::filesystem::path aligner_test_dir{get_aligner_data_dir()};
        auto ref = aligner_test_dir / "target.fq";
        return ref.string();
    }();
    return reference_file;
}

const dorado::alignment::Minimap2Options& invalid_options() {
    static const dorado::alignment::Minimap2Options result = []() {
        dorado::alignment::Minimap2Options options{dorado::alignment::dflt_options};
        options.bandwidth = options.bandwidth_long + 1;
        return options;
    }();
    return result;
}
}  // namespace

namespace dorado::alignment::index_file_access {

TEST_CASE(TEST_GROUP " constructor does not throw", TEST_GROUP) {
    REQUIRE_NOTHROW(IndexFileAccess{});
}

TEST_CASE(TEST_GROUP " load_index with invalid file return reference_file_not_found ", TEST_GROUP) {
    IndexFileAccess cut{};

    REQUIRE(cut.load_index("invalid_file_path", dflt_options, 1) ==
            IndexLoadResult::reference_file_not_found);
}

TEST_CASE(TEST_GROUP " load_index with invalid options returns validation_error", TEST_GROUP) {
    IndexFileAccess cut{};

    REQUIRE(cut.load_index(valid_reference_file(), invalid_options(), 1) ==
            IndexLoadResult::validation_error);
}

TEST_CASE(TEST_GROUP " load_index with valid arguments returns success", TEST_GROUP) {
    IndexFileAccess cut{};

    REQUIRE(cut.load_index(valid_reference_file(), dflt_options, 1) == IndexLoadResult::success);
}

SCENARIO(TEST_GROUP " Load and retrieve index files", TEST_GROUP) {
    IndexFileAccess cut{};

    GIVEN("No index loaded") {
        THEN("is_index_loaded returns false") {
            REQUIRE_FALSE(cut.is_index_loaded("blah", dflt_options));
        }
    }

    GIVEN("load_index called with valid file but invalid options") {
        cut.load_index(valid_reference_file(), invalid_options(), 1);
        THEN("is_index_loaded returns false") {
            REQUIRE_FALSE(cut.is_index_loaded(valid_reference_file(), invalid_options()));
        }
    }

    GIVEN("load_index called with valid file and default options") {
        cut.load_index(valid_reference_file(), dflt_options, 1);
        THEN("is_index_loaded returns true") {
            REQUIRE(cut.is_index_loaded(valid_reference_file(), dflt_options));
        }
        THEN("get_index returns non null index") {
            REQUIRE(cut.get_index(valid_reference_file(), dflt_options) != nullptr);
        }
        AND_GIVEN("load_index called with same file and other valid options") {
            Minimap2Options other_options{dflt_options};
            other_options.kmer_size = other_options.kmer_size + 1;
            cut.load_index(valid_reference_file(), other_options, 1);
            THEN("is_index_loaded with other options returns true") {
                REQUIRE(cut.is_index_loaded(valid_reference_file(), other_options));
            }
            THEN("get_index with other options returns a non-null index") {
                REQUIRE(cut.get_index(valid_reference_file(), other_options) != nullptr);
            }
            THEN("get_index with other options returns a different index") {
                auto default_options_index = cut.get_index(valid_reference_file(), dflt_options);
                REQUIRE(cut.get_index(valid_reference_file(), other_options) !=
                        default_options_index);
            }
        }
    }
}

TEST_CASE(TEST_GROUP " validate_options with invalid options returns false", TEST_GROUP) {
    REQUIRE_FALSE(validate_options(invalid_options()));
}

TEST_CASE(TEST_GROUP " validate_options with default options returns true", TEST_GROUP) {
    REQUIRE(validate_options(dflt_options));
}

}  // namespace dorado::alignment::index_file_access