#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "data_loader/DataLoader.h"
#include "read_pipeline/ReadPipeline.h"

#include <catch2/catch.hpp>

#define TEST_GROUP "Pod5DataLoaderTest: "

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from data dir, empty read list") {
    auto read_list = std::unordered_set<std::string>();
    CHECK(CountSinkReads(get_pod5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from single file path, empty read list") {
    auto read_list = std::unordered_set<std::string>();
    CHECK(CountSinkReads(get_single_pod5_file_path(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from data dir, no read list") {
    CHECK(CountSinkReads(get_pod5_data_dir(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from single file path, no read list") {
    CHECK(CountSinkReads(get_single_pod5_file_path(), "cpu", 1, 0, std::nullopt, {}) == 1);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from data dir, mismatched read list") {
    auto read_list = std::unordered_set<std::string>{"read_1"};
    CHECK(CountSinkReads(get_pod5_data_dir(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP
          "Test loading single-read POD5 file from single file path, mismatched read list") {
    auto read_list = std::unordered_set<std::string>{"read_1"};
    CHECK(CountSinkReads(get_single_pod5_file_path(), "cpu", 1, 0, read_list, {}) == 0);
}

TEST_CASE(TEST_GROUP "Test loading single-read POD5 file from data dir, matched read list") {
    auto read_list = std::unordered_set<std::string>{"002bd127-db82-436f-b828-28567c3d505d"};
    CHECK(CountSinkReads(get_pod5_data_dir(), "cpu", 1, 0, read_list, {}) == 1);
}

TEST_CASE(TEST_GROUP
          "Test loading single-read POD5 file from single file path, matched read list") {
    auto read_list = std::unordered_set<std::string>{"002bd127-db82-436f-b828-28567c3d505d"};
    CHECK(CountSinkReads(get_single_pod5_file_path(), "cpu", 1, 0, read_list, {}) == 1);
}

TEST_CASE(TEST_GROUP "Test calculating number of reads from pod5, read ids list.") {
    std::string data_path(get_pod5_data_dir());

    SECTION("pod5 file only, no read ids list") {
        CHECK(dorado::DataLoader::get_num_reads(data_path, std::nullopt, {}, false) == 1);
    }

    SECTION("pod5 file and read ids with 0 reads") {
        auto read_list = std::unordered_set<std::string>();
        CHECK(dorado::DataLoader::get_num_reads(data_path, read_list, {}, false) == 0);
    }
    SECTION("pod5 file and read ids with 2 reads") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("1");
        read_list.insert("2");
        CHECK(dorado::DataLoader::get_num_reads(data_path, read_list, {}, false) == 1);
    }
}

TEST_CASE(TEST_GROUP "Find sample rate from single pod5.") {
    std::string single_read_path(get_pod5_data_dir());
    CHECK(dorado::DataLoader::get_sample_rate(single_read_path, false) == 4000);
}

TEST_CASE(TEST_GROUP "Find sample rate from pod5 dir.") {
    std::string data_path(get_pod5_data_dir());
    CHECK(dorado::DataLoader::get_sample_rate(data_path, false) == 4000);
}

TEST_CASE(TEST_GROUP "Find sample rate from nested pod5.") {
    std::string data_path(get_nested_pod5_data_dir());
    CHECK(dorado::DataLoader::get_sample_rate(data_path, true) == 4000);
}

TEST_CASE(TEST_GROUP "Load data sorted by channel id.") {
    std::string data_path(get_data_dir("multi_read_pod5"));

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::DataLoader loader(*pipeline, "cpu", 1, 0, std::nullopt, {});
    loader.load_reads(data_path, true, dorado::ReadOrder::BY_CHANNEL);
    pipeline.reset();
    auto reads = ConvertMessages<dorado::SimplexReadPtr>(std::move(messages));

    int start_channel_id = -1;
    for (auto & i : reads) {
        CHECK(i->read_common.attributes.channel_number >= start_channel_id);
        start_channel_id = i->read_common.attributes.channel_number;
    }
}

TEST_CASE(TEST_GROUP "Test loading POD5 file with read ignore list") {
    std::string data_path(get_data_dir("multi_read_pod5"));

    SECTION("read ignore list with 1 read") {
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5

        CHECK(dorado::DataLoader::get_num_reads(data_path, std::nullopt, read_ignore_list, false) ==
              3);
        CHECK(CountSinkReads(data_path, "cpu", 1, 0, std::nullopt, read_ignore_list) == 3);
    }

    SECTION("same read in read_ids and ignore list") {
        auto read_list = std::unordered_set<std::string>();
        read_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5
        auto read_ignore_list = std::unordered_set<std::string>();
        read_ignore_list.insert("0007f755-bc82-432c-82be-76220b107ec5");  // read present in POD5

        CHECK(dorado::DataLoader::get_num_reads(data_path, read_list, read_ignore_list, false) ==
              0);
        CHECK(CountSinkReads(data_path, "cpu", 1, 0, read_list, read_ignore_list) == 0);
    }
}

TEST_CASE(TEST_GROUP "Test correct previous and next read ids when loaded by channel order.") {
    std::string data_path(get_data_dir("single_channel_multi_read_pod5"));

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    pipeline_desc.add_node<MessageSinkToVector>({}, 10, messages);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    dorado::DataLoader loader(*pipeline, "cpu", 1, 0, std::nullopt, {});
    loader.load_reads(data_path, true, dorado::ReadOrder::BY_CHANNEL);
    pipeline.reset();
    auto reads = ConvertMessages<dorado::SimplexReadPtr>(std::move(messages));
    std::sort(reads.begin(), reads.end(), [](auto & a, auto & b) {
        return a->read_common.start_time_ms < b->read_common.start_time_ms;
    });

    std::string prev_read_id = "";
    for (auto & i : reads) {
        CHECK(prev_read_id == i->prev_read);
        prev_read_id = i->read_common.read_id;
    }

    std::string next_read_id = "";
    for (auto i = reads.rbegin(); i != reads.rend(); i++) {
        CHECK(next_read_id == (*i)->next_read);
        next_read_id = (*i)->read_common.read_id;
    }
}
