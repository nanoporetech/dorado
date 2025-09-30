#include "cli/cli.h"
#include "data_loader/DataLoader.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "model_resolver/ModelResolver.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "utils/log_utils.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace dorado {
using namespace dorado::model_resolution;

namespace {

void print_models(bool yaml) {
    std::unordered_map<std::string_view, const std::vector<models::ModelInfo>& (*)()> all_models;
    all_models["simplex models"] = models::simplex_models;
    all_models["stereo models"] = models::stereo_models;
    all_models["modification models"] = models::modified_models;
    all_models["correction models"] = models::correction_models;
    all_models["polish models"] = models::polish_models;
    all_models["variant models"] = models::variant_models;

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

bool download_named(const models::ModelComplex& complex,
                    model_downloader::ModelDownloader& downloader) {
    using namespace dorado::models;

    if (complex.is_path_style()) {
        if (complex.get_raw() == "all") {
            std::vector<ModelInfo> models;
            const auto& all_groups = {simplex_models(),    modified_models(), stereo_models(),
                                      correction_models(), polish_models(),   variant_models()};
            for (const auto& group : all_groups) {
                for (const auto& info : group) {
                    models.push_back(info);
                }
            }

            downloader.get(models, "your");
            return true;
        }

        print_models(false);
        spdlog::error("'{}' is not a valid model name.", complex.get_raw());
        return false;
    }

    if (complex.is_named_style()) {
        std::vector<ModelInfo> models;
        models.push_back(complex.get_named_simplex_model());
        const auto& mods = complex.get_named_mods_models();
        std::copy(mods.cbegin(), mods.cend(), std::back_inserter(models));
        downloader.get(models, "your");
        return true;
    }

    if (complex.is_variant_style()) {
        throw std::logic_error("Cannot use download_named with variant model complex.");
    }

    return false;
}

bool download_variant_via_resolver(const argparse::ArgumentParser& parser) {
    const auto data = parser.get<std::string>("--data");
    const auto recursive = parser.get<bool>("--recursive");

    try {
        DataLoader::InputFiles input_files;
        if (!data.empty()) {
            input_files = DataLoader::InputFiles::search_pod5s(data, recursive);
            if (input_files.get().empty()) {
                spdlog::error("No POD5 files found in '{}' recursive:'{}'.", data, recursive);
                return false;
            }
        }

        DownloaderModelResolver resolver{
                parser.get<std::string>("--model"),
                get_models_directory(parser.get<std::string>("--models-directory")),
                input_files.get()};
        const Models models(resolver.resolve());
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        spdlog::info("Show all available models with the `--list` argument.");
        return false;
    }

    return true;
}

}  // namespace

using namespace models;

int download(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .action([&](const auto&) { ++verbosity; })
            .append();

    {
        parser.add_group("model selection");
        parser.add_argument("--model")
                .default_value(std::string("all"))
                .help("the model to download");

        auto& models_dir_arg = parser.add_argument("--models-directory")
                                       .default_value(std::string("."))
                                       .help("the directory to download the models into");
        parser.add_hidden_alias_for(models_dir_arg, "--directory");

        parser.add_argument("--data")
                .default_value(std::string())
                .help("path to POD5 data used to automatically select models");
        parser.add_argument("-r", "--recursive")
                .default_value(false)
                .implicit_value(true)
                .help("recursively scan through directories to load POD5 files");
    }
    {
        parser.add_group("model lists");
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
    }

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

    const auto models_directory =
            get_models_directory(parser.get<std::string>("--models-directory"));
    model_downloader::ModelDownloader downloader(models_directory, true);

    const auto model_arg = parser.get<std::string>("--model");
    const auto named_model = try_get_model_info(model_arg);
    if (named_model.has_value()) {
        downloader.get(named_model.value(), to_string(named_model->model_type));
    } else {
        const auto model_complex = ModelComplex::parse(model_arg);
        if (model_complex.is_variant_style()) {
            if (!download_variant_via_resolver(parser)) {
                return EXIT_FAILURE;
            }
        } else {
            if (!download_named(model_complex, downloader)) {
                return EXIT_FAILURE;
            }
        }
    }

    spdlog::info("Finished");
    return EXIT_SUCCESS;
}

}  // namespace dorado
