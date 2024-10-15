#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#include <memory>
#include <string>
#include <vector>

#define TEST_GROUP "Fast5DataLoaderTest: "

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 files") {
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, empty read list") {
    auto read_list = std::unordered_set<std::string>();
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, no read list") {
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, mismatched read list") {
    auto read_list = std::unordered_set<std::string>{"read_1"};
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, matched read list") {
    // read present in Fast5 file
    auto read_list = std::unordered_set<std::string>{"59097f00-0f1c-4fac-aea2-3c23d79b0a58"};
    CHECK(CountSinkReads(get_fast5_data_dir(), "cpu", 1, 0, read_list, {}) == 1);
}
