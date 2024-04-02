#include "TestUtils.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/ReadPipeline.h"
#include "splitter/RNAReadSplitter.h"

#include <torch/serialize.h>
// Catch2 must come after torch since both define CHECK()
#include <catch2/catch.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

#define TEST_GROUP "[RNASplitTest]"

TEST_CASE("2 subread split", TEST_GROUP) {
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

    const auto signal_path = std::filesystem::path(get_data_dir("rna_split")) / "signal.tensor";
    torch::load(read->read_common.raw_data, signal_path.string());
    read->read_common.read_tag = 42;
    read->read_common.client_info = std::make_shared<dorado::DefaultClientInfo>();

    dorado::splitter::RNASplitSettings splitter_settings;
    dorado::splitter::RNAReadSplitter splitter_node(splitter_settings);

    const auto split_res = splitter_node.split(std::move(read));
    CHECK(split_res.size() == 2);

    std::vector<uint64_t> num_samples;
    for (auto &r : split_res) {
        num_samples.push_back(r->read_common.attributes.num_samples);
    }
    CHECK(num_samples == std::vector<uint64_t>{4833, 5657});

    std::vector<uint32_t> split_points;
    for (auto &r : split_res) {
        split_points.push_back(r->read_common.split_point);
    }
    CHECK(split_points == std::vector<uint32_t>{0, 4837});
}
