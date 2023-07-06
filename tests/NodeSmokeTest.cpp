#include "MessageSinkUtils.h"
#include "decode/CPUDecoder.h"
#include "nn/CRFModel.h"
#include "nn/ModBaseRunner.h"
#include "nn/ModelRunner.h"
#include "nn/RemoraModel.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ScalerNode.h"
#include "utils/models.h"
#include "utils/parameters.h"

#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD

#include <catch2/catch.hpp>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <random>

namespace {

// Fixture for smoke testing nodes
template <typename MessageT>
class NodeSmokeTestBase {
protected:
    std::minstd_rand m_rng{Catch::rngSeed()};
    std::uniform_real_distribution<> m_dist;

    float random_between(float min, float max) {
        typename decltype(m_dist)::param_type range(min, max);
        return m_dist(m_rng, range);
    }

    auto make_test_read(std::string read_id) {
        auto read = std::make_unique<dorado::Read>();
        read->raw_data = torch::rand(random_between(100, 200));
        read->sample_rate = 5000;
        read->shift = random_between(100, 200);
        read->scale = random_between(5, 10);
        read->read_id = std::move(read_id);
        read->seq = "ACGTACGT";
        read->qstring = "********";
        read->num_trimmed_samples = random_between(10, 100);
        read->attributes.mux = 2;
        read->attributes.read_number = 12345;
        read->attributes.channel_number = 5;
        read->attributes.start_time = "2017-04-29T09:10:04Z";
        read->attributes.fast5_filename = "test.fast5";
        return read;
    }

private:
    std::size_t m_num_reads = 200;
    using ReadMutator = std::function<void(std::unique_ptr<dorado::Read>& read)>;
    ReadMutator m_read_mutator;

protected:
    void set_num_reads(std::size_t num_reads) {
        //REQUIRE_FALSE(m_sink);
        m_num_reads = num_reads;
    }

    void set_read_mutator(ReadMutator mutator) { m_read_mutator = std::move(mutator); }

    /*Sink& get_sink() {
        REQUIRE_FALSE(m_sink);
        m_sink = std::make_unique<Sink>(m_num_reads / 3);
        return *m_sink;
    }

    template <typename Node>
    void run_smoke_test(Node& node) {
        // Throw some reads at it
        for (std::size_t i = 0; i < m_num_reads; i++) {
            auto read = make_test_read("read_" + std::to_string(i));
            if (m_read_mutator) {
                m_read_mutator(read);
            }
            node.push_message(std::move(read));
        }

        // Wait for them to complete
        m_sink->wait_for_messages(m_num_reads);
    }*/
    template <class NodeType, class... Args>
    void run_smoke_test(Args&&... args) {
        dorado::PipelineDescriptor pipeline_desc;
        std::vector<dorado::Message> messages;
        auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
        pipeline_desc.add_node<NodeType>({sink}, std::forward<Args>(args)...);
        auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc));
        // Throw some reads at it.
        for (std::size_t i = 0; i < m_num_reads; i++) {
            auto read = make_test_read("read_" + std::to_string(i));
            if (m_read_mutator) {
                m_read_mutator(read);
            }
            pipeline->push_message(std::move(read));
        }
        // Wait for them to complete.
        pipeline.reset();
    }
};

using NodeSmokeTestRead = NodeSmokeTestBase<std::shared_ptr<dorado::Read>>;
using NodeSmokeTestBam = NodeSmokeTestBase<dorado::BamPtr>;

#define DEFINE_TEST(base, name) TEST_CASE_METHOD(base, "SmokeTest: " name, "[SmokeTest]")

// Not introduced until catch2 3.3.0
#ifndef SKIP
#define SKIP(msg)                                           \
    do {                                                    \
        std::cerr << "Skipping test: " << msg << std::endl; \
        return;                                             \
    } while (false)
#endif

// Wrapper around a temporary directory since one doesn't exist in the standard
struct TempDir {
    TempDir(std::filesystem::path path) : m_path(std::move(path)) {}
    ~TempDir() { std::filesystem::remove_all(m_path); }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    std::filesystem::path m_path;
};

// Download a model to a temporary directory
TempDir download_model(std::string const& model) {
    // Create a new directory to download the model to
#ifdef _WIN32
    std::filesystem::path path;
    while (true) {
        char temp[L_tmpnam];
        char const* name = std::tmpnam(temp);
        if (std::filesystem::create_directories(name)) {
            path = std::filesystem::canonical(name);
            break;
        }
    }
#else
    // macOS (rightfully) complains about tmpnam() usage, so make use of mkdtemp() on platforms that support it
    std::string temp = (std::filesystem::temp_directory_path() / "model_XXXXXXXXXX").string();
    char const* name = mkdtemp(temp.data());
    auto path = std::filesystem::canonical(name);
#endif

    // Download it
    dorado::utils::download_models(path.string(), model);
    return TempDir(std::move(path));
}

DEFINE_TEST(NodeSmokeTestRead, "ScalerNode") {
    // Scaler node expects i16 input
    set_read_mutator([](std::unique_ptr<dorado::Read>& read) {
        read->raw_data = read->raw_data.to(torch::kI16);
    });

    dorado::SignalNormalisationParams config;
    config.quantile_a = 0.2;
    config.quantile_b = 0.9;
    config.shift_multiplier = 0.51;
    config.scale_multiplier = 0.53;
    run_smoke_test<dorado::ScalerNode>(config, 2);
}

DEFINE_TEST(NodeSmokeTestRead, "BasecallerNode") {
    auto gpu = GENERATE(true, false);
    CAPTURE(gpu);

    const int kBatchTimeoutMS = 100;
    auto const& default_params = dorado::utils::default_parameters;
    char const model_name[] = "dna_r10.4.1_e8.2_400bps_fast@v4.2.0";
    auto const model_dir = download_model(model_name);
    auto const model_path = (model_dir.m_path / model_name).string();
    auto model_config = dorado::load_crf_model_config(model_path);

    // Create runners
    std::vector<dorado::Runner> runners;
    if (gpu) {
#if DORADO_GPU_BUILD
#ifdef __APPLE__
        auto caller = dorado::create_metal_caller(model_config, default_params.chunksize,
                                                  default_params.batchsize);
        for (size_t i = 0; i < default_params.num_runners; i++) {
            runners.push_back(std::make_shared<dorado::MetalModelRunner>(caller));
        }
#else   // __APPLE__
        auto devices = dorado::utils::parse_cuda_device_string("cuda:all");
        if (devices.empty()) {
            SKIP("No CUDA devices found");
        }
        for (auto const& device : devices) {
            auto caller = dorado::create_cuda_caller(model_config, default_params.chunksize,
                                                     default_params.batchsize, device);
            for (size_t i = 0; i < default_params.num_runners; i++) {
                runners.push_back(std::make_shared<dorado::CudaModelRunner>(caller));
            }
        }
#endif  // __APPLE__
#else   // DORADO_GPU_BUILD
        SKIP("Can't test GPU without DORADO_GPU_BUILD");
#endif  // DORADO_GPU_BUILD
    } else {
        const std::size_t batch_size = 128;
        runners.push_back(std::make_shared<dorado::ModelRunner<dorado::CPUDecoder>>(
                model_config, "cpu", default_params.chunksize, batch_size));
    }

    run_smoke_test<dorado::BasecallerNode>(std::move(runners),
                                           dorado::utils::default_parameters.overlap,
                                           kBatchTimeoutMS, model_name);
}

DEFINE_TEST(NodeSmokeTestRead, "ModBaseCallerNode") {
    auto gpu = GENERATE(true, false);
    CAPTURE(gpu);

    auto const& default_params = dorado::utils::default_parameters;
    char const remora_model_name[] = "dna_r10.4.1_e8.2_400bps_fast@v4.2.0_5mCG_5hmCG@v2";
    auto const remora_model_dir = download_model(remora_model_name);
    auto const remora_model = remora_model_dir.m_path / remora_model_name;

    // Add a second model into the mix
    char const remora_model_6mA_name[] = "dna_r10.4.1_e8.2_400bps_sup@v4.2.0_6mA@v2";
    auto const remora_model_6mA_dir = download_model(remora_model_6mA_name);
    auto const remora_model_6mA = remora_model_6mA_dir.m_path / remora_model_6mA_name;

    // The model stride for RemoraCaller isn't in its config so grab it separately.
    // Note: We only look at the stride of one of the models since it's not what we're
    // testing for. In theory we could hardcode the stride to any number here, but to
    // be somewhat realistic we'll use an actual one.
    char const model_name[] = "dna_r10.4.1_e8.2_400bps_fast@v4.2.0";
    auto const model_dir = download_model(model_name);
    std::size_t const model_stride =
            dorado::load_crf_model_config(model_dir.m_path / model_name).stride;

    // Create runners
    std::vector<std::unique_ptr<dorado::ModBaseRunner>> remora_runners;
    std::vector<std::string> modbase_devices;
    int batch_size = default_params.remora_batchsize;
    if (gpu) {
#if DORADO_GPU_BUILD
#ifdef __APPLE__
        // Metal falls back to CPU, which is very slow, so skip that since we're not testing anything new.
        SKIP("No metal backend for modbase");
#else   //__APPLE__
        modbase_devices = dorado::utils::parse_cuda_device_string("cuda:all");
        if (modbase_devices.empty()) {
            SKIP("No CUDA devices found");
        }
#endif  // __APPLE__
#else   // DORADO_GPU_BUILD
        SKIP("Can't test GPU without DORADO_GPU_BUILD");
#endif  // DORADO_GPU_BUILD
    } else {
        // CPU processing is very slow, so reduce the number of test reads we throw at it.
        set_num_reads(5);
        modbase_devices.push_back("cpu");
        batch_size = 8;  // reduce batch size so we're not doing work on empty entries
    }
    for (const auto& device_string : modbase_devices) {
        auto caller = dorado::create_modbase_caller({remora_model, remora_model_6mA}, batch_size,
                                                    device_string);
        for (size_t i = 0; i < default_params.remora_runners_per_caller; i++) {
            remora_runners.push_back(std::make_unique<dorado::ModBaseRunner>(caller));
        }
    }

    // ModBase node expects half input and needs a move table
    set_read_mutator([this, model_stride](std::unique_ptr<dorado::Read>& read) {
        read->raw_data = read->raw_data.to(torch::kHalf);

        read->model_stride = model_stride;
        // The move table size needs rounding up.
        size_t const move_table_size = (read->raw_data.size(0) + model_stride - 1) / model_stride;
        read->moves.resize(move_table_size);
        std::fill_n(read->moves.begin(), read->seq.size(), 1);
        // First element must be 1, the rest can be shuffled
        std::shuffle(std::next(read->moves.begin()), read->moves.end(), m_rng);
    });

    run_smoke_test<dorado::ModBaseCallerNode>(std::move(remora_runners), 2, modbase_devices.size(),
                                              model_stride, default_params.remora_batchsize);
}

DEFINE_TEST(NodeSmokeTestBam, "ReadToBamType") {
    auto emit_moves = GENERATE(true, false);
    auto rna = GENERATE(true, false);
    CAPTURE(emit_moves);
    CAPTURE(rna);

    run_smoke_test<dorado::ReadToBamType>(emit_moves, rna, 2,
                                          dorado::utils::default_parameters.methylation_threshold);
}

}  // namespace
