#include "read_pipeline/PolyACalculator.h"

#include "MessageSinkUtils.h"
#include "TestUtils.h"
#include "utils/sequence_utils.h"

#include <catch2/catch.hpp>
#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#define TEST_GROUP "[poly_a_estimator]"

namespace fs = std::filesystem;

using namespace dorado;

TEST_CASE("PolyACalculator: Test cDNA polyT tail", TEST_GROUP) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    const bool is_rna = false;
    auto demuxer = pipeline_desc.add_node<PolyACalculator>({sink}, 2, is_rna);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    auto [gt, data] = GENERATE(std::tuple<int, std::string>{70, "poly_a/r9_rev_cdna"},
                               std::tuple<int, std::string>{31, "poly_a/r10_fwd_cdna"});
    fs::path data_dir = fs::path(get_data_dir(data));
    auto seq_file = data_dir / "seq.txt";
    auto signal_file = data_dir / "signal.tensor";
    auto moves_file = data_dir / "moves.bin";
    auto read = dorado::ReadPtr::make();
    read->seq = ReadFileIntoString(seq_file.string());
    read->qstring = std::string(read->seq.length(), '~');
    read->moves = ReadFileIntoVector(moves_file.string());
    read->model_stride = 5;
    torch::load(read->raw_data, signal_file.string());
    read->read_id = "read_id";
    // Push a Read type.
    pipeline->push_message(std::move(read));

    pipeline->terminate(DefaultFlushOptions());

    CHECK(messages.size() == 1);

    auto out = std::get<ReadPtr>(std::move(messages[0]));
    CHECK(out->rna_poly_tail_length == gt);
}

TEST_CASE("PolyACalculator: Test dRNA polyT tail", TEST_GROUP) {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    const bool is_rna = true;
    auto demuxer = pipeline_desc.add_node<PolyACalculator>({sink}, 2, is_rna);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    auto [gt, data] = GENERATE(std::tuple<int, std::string>{22, "poly_a/rna002"},
                               std::tuple<int, std::string>{7, "poly_a/rna004"});
    fs::path data_dir = fs::path(get_data_dir(data));
    auto seq_file = data_dir / "seq.txt";
    auto signal_file = data_dir / "signal.tensor";
    auto moves_file = data_dir / "moves.bin";
    auto read = dorado::ReadPtr::make();
    read->seq = ReadFileIntoString(seq_file.string());
    read->qstring = std::string(read->seq.length(), '~');
    read->moves = ReadFileIntoVector(moves_file.string());
    read->model_stride = 5;
    torch::load(read->raw_data, signal_file.string());
    read->read_id = "read_id";
    // Push a Read type.
    pipeline->push_message(std::move(read));

    pipeline->terminate(DefaultFlushOptions());

    CHECK(messages.size() == 1);

    auto out = std::get<ReadPtr>(std::move(messages[0]));
    CHECK(out->rna_poly_tail_length == gt);
}
