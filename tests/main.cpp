
#include "torch_utils/torch_utils.h"
#include "utils/locale_utils.h"

#include <nvtx3/nvtx3.hpp>
#include <torch/utils.h>

// Catch must come last so we can undo torch defining CHECK.
#undef CHECK
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

int main(int argc, char* argv[]) {
    // global setup...

    dorado::utils::ensure_user_locale_may_be_set();

    dorado::utils::make_torch_deterministic();
    torch::set_num_threads(1);

    // Initialize NVTX first before any tests are run. This is
    // needed because the NVTX initialization is not thread safe,
    // and some of the tests launch multiple threads each
    // of which trigger an NVTX init which causes the thread
    // sanitizers to fail.
    { nvtx3::scoped_range loop{__func__}; }

    int result = Catch::Session().run(argc, argv);

    return result;
}
