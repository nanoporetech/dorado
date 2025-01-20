#include "TestUtils.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/ReadPipeline.h"
#include "splitter/RNAReadSplitter.h"

#include <catch2/catch_test_macros.hpp>
#include <torch/serialize.h>

#include <cstdint>
#include <filesystem>
#include <vector>

#define TEST_GROUP "[RNASplitTest]"

CATCH_TEST_CASE("2 subread split", TEST_GROUP) {
    auto read = std::make_unique<dorado::SimplexRead>();
    read->range = 0;
    read->read_common.sample_rate = 4000;
    read->read_common.read_id = "1ebbe001-d735-4191-af79-bee5a2fca7dd";
    read->read_common.num_trimmed_samples = 0;
    read->read_common.attributes.read_number = 57296;
    read->read_common.attributes.channel_number = 2207;
    read->read_common.attributes.mux = 4;
    read->read_common.attributes.start_time = "2023-08-11T02:56:14.296+00:00";
    read->read_common.attributes.num_samples = 10494;
    read->read_common.scaling_method = "test";

    const auto signal_path = std::filesystem::path(get_data_dir("rna_split")) / "signal.tensor";
    torch::load(read->read_common.raw_data, signal_path.string());
    read->read_common.read_tag = 42;
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    dorado::splitter::RNASplitSettings splitter_settings;
    dorado::splitter::RNAReadSplitter splitter_node(splitter_settings);

    const auto split_res = splitter_node.split(std::move(read));
    CATCH_REQUIRE(split_res.size() == 2);

    CATCH_CHECK(split_res[0]->read_common.attributes.num_samples == 4833);
    CATCH_CHECK(split_res[0]->read_common.split_point == 0);
    CATCH_CHECK(split_res[0]->read_common.scaling_method == "test");

    CATCH_CHECK(split_res[1]->read_common.attributes.num_samples == 5657);
    CATCH_CHECK(split_res[1]->read_common.split_point == 4837);
    CATCH_CHECK(split_res[1]->read_common.scaling_method == "test");
}
