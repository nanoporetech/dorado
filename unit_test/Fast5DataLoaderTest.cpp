#include <catch2/catch.hpp>

#include "data_loader/Fast5DataLoader.h"
#include "TestUtils.h"

#define TEST_GROUP "Fast5DataLoaderTest: "

class MockSink : public ReadSink {
public:
    size_t get_read_count() { return m_reads.size(); }
};


TEST_CASE( TEST_GROUP "Test loading single-read Fast5 files" ) {
    // Create a mock sink for testing output of reads
    MockSink mock_sink;

    std::string data_path(get_data_dir());
    Fast5DataLoader loader(mock_sink, "cpu");
    loader.load_reads(data_path);

    REQUIRE(mock_sink.get_read_count() == 1);
}