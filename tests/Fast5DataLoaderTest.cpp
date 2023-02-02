#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#include <memory>

#define TEST_GROUP "Fast5DataLoaderTest: "

class MockSink : public dorado::ReadSink {
public:
    MockSink() : ReadSink(1000) {}
    size_t get_read_count();
};

size_t MockSink::get_read_count() {
    size_t read_count = 0;
    std::shared_ptr<dorado::Read> read;
    while (m_work_queue.try_pop(read))
        ++read_count;
    return read_count;
}

TEST_CASE(TEST_GROUP "Test loading single-read Fast5 files") {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_data_dir());
    dorado::DataLoader loader(mock_sink, "cpu", 1);
    // Note: Upon completion load_reads calls terminate on its sink, so the
    // the loop in get_reads_count will finish as soon as the queue is empty.
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}