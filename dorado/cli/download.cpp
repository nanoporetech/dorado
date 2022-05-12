#define CPPHTTPLIB_OPENSSL_SUPPORT

#include <iostream>
#include <argparse.hpp>
#include <filesystem>

#include "models.h"
#include "Version.h"

#include "httplib.h"
#include "elzip/elzip.hpp"

namespace fs = std::filesystem;


int download(int argc, char *argv[]) {

    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    parser.add_argument("--model")
            .default_value(std::string("all"))
            .help("the model to download");

    parser.add_argument("--directory")
           .default_value(std::string("."))
           .help("the directory to download the models into");

    parser.add_argument("--list")
            .default_value(false)
            .implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    auto list = parser.get<bool>("--list");
    auto selected_model = parser.get<std::string>("--model");
    auto directory = fs::path(parser.get<std::string>("--directory"));

    if (list) {
        std::cerr << "> basecaller models" << std::endl;
        for (const auto& [model, _] : basecaller::models) {
            std::cerr << " - " << model << std::endl;
        }
        return 0;
    }

    if (selected_model != "all" && basecaller::models.find(selected_model) == basecaller::models.end()) {
        std::cerr << "> error: '" << selected_model << "' is not a valid model" << std::endl;
        std::cerr << "> basecaller models" << std::endl;
        for (const auto& [model, _] : basecaller::models) {
            std::cerr << " - " << model << std::endl;
        }
        return 1;
    }

    httplib::Client http(basecaller::URL_ROOT);
    http.set_follow_location(true);

    for (const auto& [model, url] : basecaller::models) {
        if (selected_model == "all" || selected_model == model) {
            std::cerr << " - downloading " << model;
            auto res = http.Get(url.c_str());
            std::cout << " [" << res->status << "]" << std::endl;
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
