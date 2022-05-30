#include <catch2/catch.hpp>

#include <filesystem>
#include <iostream>

static std::string get_data_dir() {
    const std::string data_path = "./tests/data";

    if (!std::filesystem::exists(data_path)) {
        std::cerr << "Datapath " << data_path
                  << " does not exist, exiting.\n"
                     "Unit tests must be run from the build working directory inside dorado"
                  << std::endl;
        exit(1);
    }
    return data_path;
}
