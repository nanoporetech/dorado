#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#include <memory>

#define TEST_GROUP "Fast5DataLoaderTest: "

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

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 files") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_fast5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1);
    // Note: Upon completion load_reads calls terminate on its sink, so the
    // the loop in get_reads_count will finish as soon as the queue is empty.
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, empty read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    std::string data_path(get_fast5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, mismatched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("read_1");
    std::string data_path(get_fast5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 file, matched read list") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    auto read_list = std::unordered_set<std::string>();
    read_list.insert("59097f00-0f1c-4fac-aea2-3c23d79b0a58");  // read present in Fast5 file
    std::string data_path(get_fast5_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1, 0, read_list);
    loader.load_reads(data_path, false);

    REQUIRE(mock_sink.get_read_count() == 1);
}