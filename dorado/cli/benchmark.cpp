#include "../utils/tensor_utils.h"
#include "dorado_version.h"

#include <ATen/ATen.h>
#include <argparse.hpp>

#include <chrono>
#include <iostream>

namespace dorado {

int benchmark(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        std::cerr << parser;
        std::exit(1);
    }

    std::vector<size_t> sizes{1000, 1000, 2000, 3000, 4000, 10000, 100000, 1000000, 10000000};

    for (auto n : sizes) {
        std::cerr << "samples : " << n << '\n';

        // generate some input
        auto x = at::randint(0, 2047, n);
        auto q = at::tensor({0.2, 0.9}, {at::ScalarType::Float});

        // torch::quantile
        auto start = std::chrono::system_clock::now();
        auto res = at::quantile(x, q);
        auto end = std::chrono::system_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "torch:quant  "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << '\n';

        // nth_element
        start = std::chrono::system_clock::now();
        res = utils::quantile(x, q);
        end = std::chrono::system_clock::now();

        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "nth_element  "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << '\n';

        x = x.to(at::ScalarType::Short);

        // counting
        start = std::chrono::system_clock::now();
        res = utils::quantile_counting(x, q);
        end = std::chrono::system_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cerr << "counting     "
                  << " q20=" << res[0].item<int>() << " q90=" << res[1].item<int>() << " "
                  << duration << "us" << '\n'
                  << '\n';
    }

    return 0;
}

}  // namespace dorado
