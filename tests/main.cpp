#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
    // global setup...

    torch::set_num_threads(1);
    int result = Catch::Session().run(argc, argv);

    // global clean-up...

    return result;
}
