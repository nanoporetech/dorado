
#include "torch_utils/torch_utils.h"
#include "utils/locale_utils.h"

#include <c10/core/InferenceMode.h>
#include <catch2/catch_session.hpp>
#include <nvtx3/nvtx3.hpp>

int main(int argc, char* argv[]) {
    // global setup...

    dorado::utils::ensure_user_locale_may_be_set();

    dorado::utils::initialise_torch();
    dorado::utils::make_torch_deterministic();

    // Initialize NVTX first before any tests are run. This is
    // needed because the NVTX initialization is not thread safe,
    // and some of the tests launch multiple threads each
    // of which trigger an NVTX init which causes the thread
    // sanitizers to fail.
    {
        nvtx3::scoped_range loop{__func__};
    }

    // Only inference during tests or benchmarks.
    c10::InferenceMode inference_mode_guard;

    int result = Catch::Session().run(argc, argv);

    return result;
}
