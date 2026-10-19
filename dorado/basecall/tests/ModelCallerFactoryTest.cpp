#include "basecall/ModelCallerFactory.h"

#include "TestUtils.h"
#include "caller/ModelCallerSelection.h"
#include "config/BasecallModelConfig.h"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#define CUT_TAG "[ModelCallerFactory]"

namespace {

dorado::config::BasecallModelConfig load_config(const char* model_name) {
    return dorado::config::load_model_config(std::filesystem::path(get_data_dir("model_configs")) /
                                             model_name);
}

}  // namespace

CATCH_TEST_CASE(CUT_TAG " exposes accelerator batch granularity without concrete callers",
                CUT_TAG) {
    const auto lstm_config = load_config("dna_r10.4.1_e8.2_400bps_hac@v4.2.0");
    const auto tx_config = load_config("dna_r10.4.1_e8.2_400bps_sup@v5.0.0");

#if DORADO_CUDA_BUILD
    CATCH_CHECK(dorado::basecall::model_caller_batch_size_granularity("cuda:0", lstm_config) == 64);
    CATCH_CHECK(dorado::basecall::model_caller_batch_size_granularity("cuda:0", tx_config) == 32);
#elif DORADO_METAL_BUILD
    CATCH_CHECK(dorado::basecall::model_caller_batch_size_granularity("metal", lstm_config) == 48);
    CATCH_CHECK(dorado::basecall::model_caller_batch_size_granularity("metal", tx_config) == 8);
#endif
}

CATCH_TEST_CASE(CUT_TAG " rejects accelerator planning for CPU", CUT_TAG) {
    const auto config = load_config("dna_r10.4.1_e8.2_400bps_hac@v4.2.0");

    CATCH_CHECK_THROWS(dorado::basecall::model_caller_batch_size_granularity("cpu", config));
    CATCH_CHECK_THROWS(dorado::basecall::model_caller_max_safe_batch_size("cpu", 1.f, config));
    CATCH_CHECK_THROWS(dorado::basecall::model_caller_memory_limit("cpu", 1.f));
}

CATCH_TEST_CASE(CUT_TAG " selects one Metal implementation for each model family", CUT_TAG) {
    using dorado::basecall::MetalModelImplementation;
    using dorado::basecall::select_metal_model_implementation;

    dorado::config::BasecallModelConfig lstm_config;
    CATCH_CHECK(select_metal_model_implementation(lstm_config) == MetalModelImplementation::lstm);

    auto flstm_config = lstm_config;
    flstm_config.lstm_inner_dim = 128;
    CATCH_CHECK(select_metal_model_implementation(flstm_config) == MetalModelImplementation::mps);

    const auto tx_config = load_config("dna_r10.4.1_e8.2_400bps_sup@v5.0.0");
    CATCH_CHECK(select_metal_model_implementation(tx_config) == MetalModelImplementation::mps);
}
