#include "../utils/tensor_utils.h"
#include "Version.h"
#include "torch/torch.h"

#include <argparse.hpp>

#include <chrono>
#include <iostream>

int benchmark(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::cerr << "benchmarker.." << std::endl;

    std::vector<size_t> sizes{1000, 1000, 2000, 3000, 4000, 10000, 100000, 1000000, 10000000};

    for (auto n : sizes) {
        // generate some input
        auto x = torch::randint(0, 2048, n);
        auto q = torch::tensor({0.2, 0.9}, {torch::kFloat32});

        // nth_element
        auto start = std::chrono::system_clock::now();
        auto res = ::utils::quantile(x, q);
        auto end = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "nth_element n=" << n << " " << duration << "us q20=";
        std::cerr << res[0].item<int>() << " q90=" << res[1].item<int>() << std::endl;

        x = x.to(torch::kInt);

        // radix sort
        start = std::chrono::system_clock::now();
        res = ::utils::quantile_counting(x, 0, 2047, q);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "counting    n=" << n << " " << duration << "us q20=";
        std::cerr << res[0].item<int>() << " q90=" << res[1].item<int>() << std::endl;
    }

    return 0;
}
