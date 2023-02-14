#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "Pod5DataLoaderTest: "

class MockSink : public dorado::ReadSink {
public:
    MockSink() : ReadSink(1000) {}
    size_t get_read_count() { return m_reads.size(); }
};

TEST_CASE(TEST_GROUP "Test loading single-read POD5 files") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1);
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, empty read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, mismatched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("read_1");
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, matched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("002bd127-db82-436f-b828-28567c3d505d");  // read present in POD5
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}