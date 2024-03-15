#include "Version.h"
#include "cli/cli_utils.h"
#include "data_loader/ModelFinder.h"
#include "models/kits.h"
#include "models/models.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace dorado {

namespace {

void print_models(bool yaml) {
    std::unordered_map<std::string_view, const std::vector<models::ModelInfo>& (*)()> all_models;
    all_models["simplex models"] = models::simplex_models;
    all_models["stereo models"] = models::stereo_models;
    all_models["modification models"] = models::modified_models;

    if (yaml) {
        for (const auto& [type, models] : all_models) {
            std::cout << type << ":\n";
            for (const auto& model_info : models()) {
                std::cout << "  - \"" << model_info.name << "\"\n";
            }
        }
    } else {
        for (const auto& [type, models] : all_models) {
            spdlog::info("> {}", type);
            for (const auto& model_info : models()) {
                spdlog::info(" - {}", model_info.name);
            }
        }
    }
}

using namespace dorado::models;

std::vector<std::string> get_model_names(const ModelSelection& model_selection,
                                         const fs::path& data,
                                         bool recursive) {
    std::vector<std::string> models;

    const auto model_arg = model_selection.raw;
    if (model_selection.is_path()) {
        if (model_arg == "all") {
            const auto& all_groups = {simplex_model_names(), modified_model_names(),
                                      stereo_model_names()};
            for (const auto& group : all_groups) {
                for (const auto& model_name : group) {
                    models.push_back(model_name);
                }
            }
            return models;
        }

        if (!is_valid_model(model_selection.raw)) {
            spdlog::error("'{}' is not a valid model", model_selection.raw);
            print_models(false);
            std::exit(EXIT_FAILURE);
        }

        models.push_back(model_arg);
        return models;
    }

    if (data.empty() && model_selection.has_model_variant()) {
        spdlog::error("Must set --data when using automatic model detection");
        std::exit(EXIT_FAILURE);
    }

    if (!data.empty()) {
        const auto model_finder = cli::model_finder(model_selection, data, recursive, true);

        try {
            if (model_selection.has_model_variant()) {
                models.push_back(model_finder.get_simplex_model_name());
            }
            // If user made a mods selection get it, otherwise get all mods
            const auto mm = model_selection.has_mods_variant()
                                    ? model_finder.get_mods_model_names()
                                    : model_finder.get_mods_for_simplex_model();
            models.insert(models.end(), mm.begin(), mm.end());
        } catch (std::exception& e) {
            spdlog::error(e.what());
            std::exit(EXIT_FAILURE);
        }

        try {
            models.push_back(model_finder.get_stereo_model_name());
        } catch (std::exception& e) {
            spdlog::debug(e.what());
        }
    }
    return models;
}

}  // namespace

using namespace models;

int download(int argc, char* argv[]) {
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

    parser.add_argument("--data")
            .default_value(std::string())
            .help("path to POD5 data used to automatically select models");
    parser.add_argument("-r", "--recursive")
            .default_value(false)
            .implicit_value(true)
            .help("recursively scan through directories to load POD5 files");
    parser.add_argument("--overwrite")
            .default_value(false)
            .implicit_value(true)
            .help("overwrite existing models if they already exist");

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
        return EXIT_FAILURE;
    }

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto list = parser.get<bool>("--list");
    auto list_yaml = parser.get<bool>("--list-yaml");

    if (list || list_yaml) {
        print_models(list_yaml);
        return EXIT_SUCCESS;
    }

    const auto model_arg = parser.get<std::string>("--model");
    const auto data = parser.get<std::string>("--data");
    const auto recursive = parser.get<bool>("--recursive");

    const auto model_selection = cli::parse_model_argument(model_arg);
    const auto model_names = get_model_names(model_selection, data, recursive);

    const auto downloads_path = utils::get_downloads_path(parser.get<std::string>("--directory"));

    const auto overwrite = parser.get<bool>("--overwrite");
    for (auto& name : model_names) {
        if (!overwrite && fs::exists(downloads_path / name)) {
            spdlog::info(" - existing model found: {}", name);
        } else if (!models::download_models(downloads_path.string(), name)) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

}  // namespace dorado
