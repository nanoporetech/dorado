#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "cli/model_resolution.h"
#include "dorado_version.h"
#include "file_info/file_info.h"
#include "model_downloader/model_downloader.h"
#include "models/kits.h"
#include "models/models.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"

#include <argparse/argparse.hpp>
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
    all_models["correction models"] = models::correction_models;
    all_models["polish models"] = models::polish_models;

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

std::vector<ModelInfo> get_model_infos(const ModelComplex& model_complex,
                                       const fs::path& data,
                                       bool recursive) {
    std::vector<ModelInfo> models;

    const auto model_arg = model_complex.raw;
    if (model_complex.is_path()) {
        if (model_arg == "all") {
            const auto& all_groups = {simplex_models(), modified_models(), stereo_models(),
                                      correction_models(), polish_models()};
            for (const auto& group : all_groups) {
                for (const auto& info : group) {
                    models.push_back(info);
                }
            }
            return models;
        }

        if (!is_valid_model(model_complex.raw)) {
            spdlog::error("'{}' is not a valid model", model_complex.raw);
            print_models(false);
            std::exit(EXIT_FAILURE);
        }

        models.push_back(models::get_model_info(model_arg));
        return models;
    }

    if (data.empty() && model_complex.has_model_variant()) {
        spdlog::error("Must set --data when using automatic model detection");
        std::exit(EXIT_FAILURE);
    }

    if (!data.empty()) {
        const auto folder_entries = utils::fetch_directory_entries(data.u8string(), recursive);
        const auto chemisty = file_info::get_unique_sequencing_chemistry(folder_entries);
        auto model_search = models::ModelComplexSearch(model_complex, chemisty, true);

        try {
            if (model_complex.has_model_variant()) {
                models.push_back(model_search.simplex());
            }
            // If user made a mods selection get it, otherwise get all mods
            const auto mm = model_complex.has_mods_variant() ? model_search.mods()
                                                             : model_search.simplex_mods();
            models.insert(models.end(), mm.begin(), mm.end());
        } catch (std::exception& e) {
            spdlog::error(e.what());
            std::exit(EXIT_FAILURE);
        }

        try {
            models.push_back(model_search.stereo());
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

    auto& models_dir_arg = parser.add_argument("--models-directory")
                                   .default_value(std::string("."))
                                   .help("the directory to download the models into");
    parser.add_hidden_alias_for(models_dir_arg, "--directory");

    parser.add_argument("--list").default_value(false).implicit_value(true).help(
            "list the available models for download");
    parser.add_argument("--list-yaml")
            .help("list the available models for download, as yaml, to stdout")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--list-structured")
            .help("list the available models in a structured format, as yaml, to stdout")
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
    auto list_structured = parser.get<bool>("--list-structured");

    if (list || list_yaml) {
        print_models(list_yaml);
        return EXIT_SUCCESS;
    }

    if (list_structured) {
        std::cout << models::get_supported_model_info("") << '\n';
        return EXIT_SUCCESS;
    }

    const auto model_arg = parser.get<std::string>("--model");
    const auto data = parser.get<std::string>("--data");
    const auto recursive = parser.get<bool>("--recursive");

    const auto model_complex = model_resolution::parse_model_argument(model_arg);
    const auto model_infos = get_model_infos(model_complex, data, recursive);

    std::filesystem::path models_directory;
    auto models_directory_opt = model_resolution::get_models_directory(parser);
    if (!models_directory_opt.has_value()) {
        models_directory = std::filesystem::path(parser.get<std::string>("--models-directory"));
    } else {
        models_directory = models_directory_opt.value();
    }

    auto downloader = model_downloader::ModelDownloader(models_directory);

    const auto overwrite = parser.get<bool>("--overwrite");
    for (auto& info : model_infos) {
        auto new_model_path = models_directory / info.name;
        if (fs::exists(new_model_path)) {
            if (!overwrite) {
                spdlog::info(" - found existing model: '{}'", info.name);
                spdlog::debug(" - model found at: '{}'", fs::canonical(new_model_path).u8string());
                continue;
            }
            spdlog::debug(" - deleting existing model: {} at: '{}'", info.name,
                          fs::canonical(new_model_path).u8string());
            fs::remove_all(new_model_path);
        }
        try {
            const auto actual_path = downloader.get(info, "your");
            spdlog::debug(" - downloaded model: '{}' into '{}'", info.name,
                          fs::canonical((actual_path)).u8string());
        } catch (const std::exception& e) {
            spdlog::debug("downloader exception: {}", e.what());
            spdlog::error("Failed to download model: {}", info.name);
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}

}  // namespace dorado
