#define CPPHTTPLIB_OPENSSL_SUPPORT

#include "Version.h"
#include "elzip/elzip.hpp"
#include "httplib.h"
#include "models.h"

#include <argparse.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int download(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    parser.add_argument("--model").default_value(std::string("all")).help("the model to download");

    parser.add_argument("--directory")
            .default_value(std::string("."))
            .help("the directory to download the models into");

    parser.add_argument("--list").default_value(false).implicit_value(true).help(
            "list the available models for download");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    auto list = parser.get<bool>("--list");
    auto selected_model = parser.get<std::string>("--model");
    auto directory = fs::path(parser.get<std::string>("--directory"));
    auto permissions = fs::status(directory).permissions();

    auto print_models = [] {
        std::cerr << "> basecaller models" << std::endl;
        for (const auto& [model, _] : basecaller::models) {
            std::cerr << " - " << model << std::endl;
        }
    };

    if (list) {
        print_models();
        return 0;
    }

    if (selected_model != "all" &&
        basecaller::models.find(selected_model) == basecaller::models.end()) {
        std::cerr << "> error: '" << selected_model << "' is not a valid model" << std::endl;
        print_models();
        return 1;
    }

    if (!fs::exists(directory)) {
        try {
            fs::create_directories(directory);
        } catch (std::filesystem::filesystem_error const& e) {
            std::cerr << "> error: " << e.code().message() << std::endl;
            return 1;
        }
    }

    std::ofstream tmp(directory / "tmp");
    tmp << "test";
    tmp.close();

    if (tmp.fail()) {
        std::cerr << "> error: insufficient permissions to download models into " << directory
                  << std::endl;
        return 1;
    }

    httplib::Client http(basecaller::URL_ROOT);
    http.set_follow_location(true);

    for (const auto& [model, url] : basecaller::models) {
        if (selected_model == "all" || selected_model == model) {
            std::cerr << " - downloading " << model;
            auto res = http.Get(url.c_str());
            std::cerr << " [" << res->status << "]" << std::endl;
            fs::path archive(directory / (model + ".zip"));
            std::ofstream ofs(archive.string());
            ofs << res->body;
            ofs.close();
            elz::extractZip(archive, directory);
            fs::remove(archive);
        }
    }

    return 0;
}
