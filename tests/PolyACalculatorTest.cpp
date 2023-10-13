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

struct TestCase {
    int estimated_bases = 0;
    std::string test_dir;
    PolyACalculator::ModelType model_type;
};

TEST_CASE("PolyACalculator: Test polyT tail estimation", TEST_GROUP) {
    auto [gt, data, model_type] =
            GENERATE(TestCase{70, "poly_a/r9_rev_cdna", PolyACalculator::ModelType::DNA},
                     TestCase{31, "poly_a/r10_fwd_cdna", PolyACalculator::ModelType::DNA},
                     TestCase{22, "poly_a/rna002", PolyACalculator::ModelType::RNA002},
                     TestCase{7, "poly_a/rna004", PolyACalculator::ModelType::RNA004});

    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    auto estimator = pipeline_desc.add_node<PolyACalculator>({sink}, 2, model_type);

    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));

    fs::path data_dir = fs::path(get_data_dir(data));
    auto seq_file = data_dir / "seq.txt";
    auto signal_file = data_dir / "signal.tensor";
    auto moves_file = data_dir / "moves.bin";
    auto read = std::make_unique<SimplexRead>();
    read->read_common.seq = ReadFileIntoString(seq_file.string());
    read->read_common.qstring = std::string(read->read_common.seq.length(), '~');
    read->read_common.moves = ReadFileIntoVector(moves_file.string());
    read->read_common.model_stride = 5;
    torch::load(read->read_common.raw_data, signal_file.string());
    read->read_common.read_id = "read_id";
    // Push a Read type.
    pipeline->push_message(std::move(read));

    pipeline->terminate(DefaultFlushOptions());

    CHECK(messages.size() == 1);

    auto out = std::get<SimplexReadPtr>(std::move(messages[0]));
    CHECK(out->read_common.rna_poly_tail_length == gt);
}
