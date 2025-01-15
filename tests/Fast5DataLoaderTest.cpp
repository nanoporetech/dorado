#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "read_pipeline/ReadPipeline.h"

#include <memory>
#include <string>
#include <vector>

// Catch must come last so we can undo torch defining CHECK.
#undef CHECK
#include <catch2/catch_all.hpp>

#define TEST_GROUP "[dorado::DataLoader::fast5]"

TEST_CASE(TEST_GROUP " Test loading single-read Fast5 files", TEST_GROUP) {
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP " Test loading single-read Fast5 file, empty read list", TEST_GROUP) {
    auto read_list = std::unordered_set<std::string>();
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP " Test loading single-read Fast5 file, no read list", TEST_GROUP) {
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP " Test loading single-read Fast5 file, mismatched read list", TEST_GROUP) {
    auto read_list = std::unordered_set<std::string>{"read_1"};
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP " Test loading single-read Fast5 file, matched read list", TEST_GROUP) {
    // read present in Fast5 file
    auto read_list = std::unordered_set<std::string>{"59097f00-0f1c-4fac-aea2-3c23d79b0a58"};
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 1);
}
