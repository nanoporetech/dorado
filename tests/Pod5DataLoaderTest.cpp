#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "Pod5DataLoaderTest: "

namespace {

class MockSink : public dorado::MessageSink {
public:
    MockSink() : MessageSink(1000) {}
    size_t get_read_count();
};

size_t MockSink::get_read_count() {
    size_t read_count = 0;
    dorado::Message read;
    while (m_work_queue.try_pop(read))
        ++read_count;
    return read_count;
}

}  // namespace

TEST_CASE(TEST_GROUP "Test loading single-read POD5 files") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, empty read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, no read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, std::nullopt);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, mismatched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("read_1");
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file, matched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("002bd127-db82-436f-b828-28567c3d505d");  // read present in POD5
    std::string data_path(get_pod5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test calculating number of reads from pod5, read ids list.") {
    // Create a mock sink for testing output of reads
    std::string data_path(get_pod5_data_dir());

    SECTION("pod5 file only, no read ids list") {
        CHECK(dorado::DataLoader::get_num_reads(data_path, std::nullopt) == 1);
    }

    SECTION("pod5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CHECK(dorado::DataLoader::get_num_reads(data_path, read_list) == 0);
    }
    SECTION("pod5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CHECK(dorado::DataLoader::get_num_reads(data_path, read_list) == 1);
    }
}

TEST_CASE(TEST_GROUP "Find sample rate from pod5.") {
    // Create a mock sink for testing output of reads
    std::string data_path(get_pod5_data_dir());

    CHECK(dorado::DataLoader::get_sample_rate(data_path).value() == 4000);
}
