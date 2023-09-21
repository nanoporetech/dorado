#define CATCH_CONFIG_RUNNER

#include "cli/cli_utils.h"

#include <catch2/catch.hpp>
#include <nvtx3/nvtx3.hpp>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
    // global setup...

    dorado::cli::make_torch_deterministic();
    torch::set_num_threads(1);

    // Initialize NVTX first before any tests are run. This is
    // needed because the NVTX initialization is not thread safe,
    // and some of the tests launch multiple threads each
    // of which trigger an NVTX init which causes the thread
    // sanitizers to fail.
    { nvtx3::scoped_range loop{__func__}; }

    int result = Catch::Session().run(argc, argv);

    // global clean-up...

    return result;
}
