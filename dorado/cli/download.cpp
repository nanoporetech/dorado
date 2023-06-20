#include "Version.h"
#include "utils/log_utils.h"
#include "utils/models.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace dorado {

int download(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("--model").default_value(std::string("all")).help("the model to download");

    parser.add_argument("--directory")
            .default_value(std::string("."))
            .help("the directory to download the models into");

    parser.add_argument("--list").default_value(false).implicit_value(true).help(
            "list the available models for download");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetDebugLogging();
    }

    auto list = parser.get<bool>("--list");
    auto selected_model = parser.get<std::string>("--model");
    auto directory = fs::path(parser.get<std::string>("--directory"));
    auto permissions = fs::status(directory).permissions();

    auto print_models = [] {
        spdlog::info("> simplex models");
        for (const auto& model : simplex::models) {
            spdlog::info(" - {}", model);
        }
        spdlog::info("> stereo models");
        for (const auto& model : stereo::models) {
            spdlog::info(" - {}", model);
        }
        spdlog::info("> modification models");
        for (const auto& model : modified::models) {
            spdlog::info(" - {}", model);
        }
    };

    if (list) {
        print_models();
        return 0;
    }

    if (!utils::is_valid_model(selected_model)) {
        spdlog::error("> error: '{}' is not a valid model", selected_model);
        print_models();
        return 1;
    }

    if (!fs::exists(directory)) {
        try {
            fs::create_directories(directory);
        } catch (std::filesystem::filesystem_error const& e) {
            spdlog::error("> error: {}", e.code().message());
            return 1;
        }
    }

    std::ofstream tmp(directory / "tmp");
    tmp << "test";
    tmp.close();

    if (tmp.fail()) {
        spdlog::error("> error: insufficient permissions to download models into {}",
                      std::string(directory.u8string()));
        return 1;
    }

    try {
        fs::remove(directory / "tmp");
    } catch (std::filesystem::filesystem_error const& e) {
        std::cerr << "> error: " << e.code().message() << std::endl;
        return 1;
    }

    utils::download_models(directory.string(), selected_model);

    return 0;
}

}  // namespace dorado
