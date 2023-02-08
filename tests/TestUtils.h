#include <catch2/catch.hpp>

#include <filesystem>
#include <iostream>

static std::string get_fast5_data_dir() {
    const std::string data_path = "./tests/data/fast5";

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the build working directory inside dorado"
                  << std::endl;
        exit(1);
    }
    return data_path;
}

static std::string get_pod5_data_dir() {
    const std::string data_path = "./tests/data/pod5";

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the build working directory inside dorado"
                  << std::endl;
        exit(1);
    }
    return data_path;
}

static std::string get_stereo_data_dir() {
    const std::string data_path = "./tests/data/stereo";

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the build working directory inside dorado"
                  << std::endl;
        exit(1);
    }
    return data_path;
}
