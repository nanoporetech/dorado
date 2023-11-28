#include "Version.h"
#include "models/models.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace dorado {

namespace {

void print_models(bool yaml) {
    std::unordered_map<std::string_view, const models::ModelMap& (*)()> all_models;
    all_models["simplex models"] = models::simplex_models;
    all_models["stereo models"] = models::stereo_models;
    all_models["modification models"] = models::modified_models;

    if (yaml) {
        for (const auto& [type, models] : all_models) {
            std::cout << type << ":\n";
            for (const auto& [model, info] : models()) {
                std::cout << "  - \"" << model << "\"\n";
            }
        }
    } else {
        for (const auto& [type, models] : all_models) {
            spdlog::info("> {}", type);
            for (const auto& [model, info] : models()) {
                spdlog::info(" - {}", model);
            }
        }
    }
}

}  // namespace

int download(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("--model").default_value(std::string("all")).help("the model to download");

    parser.add_argument("--directory")
            .default_value(std::string("."))
            .help("the directory to download the models into");

    parser.add_argument("--list").default_value(false).implicit_value(true).help(
            "list the available models for download");
    parser.add_argument("--list-yaml")
            .help("list the available models for download, as yaml, to stdout")
            .default_value(false)
            .implicit_value(true);

    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto list = parser.get<bool>("--list");
    auto list_yaml = parser.get<bool>("--list-yaml");
    auto selected_model = parser.get<std::string>("--model");
    auto directory = fs::path(parser.get<std::string>("--directory"));

    if (list || list_yaml) {
        print_models(list_yaml);
        return 0;
    }

    if (!models::is_valid_model(selected_model)) {
        spdlog::error("> error: '{}' is not a valid model", selected_model);
        print_models(false);
        return 1;
    }

    if (!fs::exists(directory)) {
        try {
            fs::create_directories(directory);
        } catch (const std::filesystem::filesystem_error& e) {
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
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "> error: " << e.code().message() << std::endl;
        return 1;
    }

    return models::download_models(directory.string(), selected_model) ? EXIT_SUCCESS
                                                                       : EXIT_FAILURE;
}

}  // namespace dorado
