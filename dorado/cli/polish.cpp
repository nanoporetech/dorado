#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/features.h"
#include "polish/medaka_bamiter.h"
#include "polish/medaka_counts.h"
#include "polish/model.h"
#include "torch_utils/auto_detect_device.h"
#include "utils/arg_parse_ext.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"

#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

namespace dorado {

namespace {

using ParserPtr = std::unique_ptr<utils::arg_parse::ArgParser>;

/// \brief All options for this tool.
struct Options {
    // Positional parameters.
    std::filesystem::path in_aln_bam_fn;
    std::filesystem::path in_draft_fastx_fn;
    std::filesystem::path out_consensus_fn;

    // Optional parameters.
    std::filesystem::path model_path;
    int32_t verbosity = 0;
    int32_t threads = 1;
    int32_t infer_threads = 1;
    std::string device;
    int32_t batch_size = 128;
    int32_t window_size = 10000;
    int32_t window_overlap = 1000;
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado consensus");

    parser->visible.add_description("Consensus tool for polishing draft assemblies");

    {
        // Positional arguments group
        parser->visible.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser->visible.add_argument("in_draft_fastx").help("Draft assembly for polishing");
        parser->visible.add_argument("out_consensus").help("Output consensus FASTA file.");
    }
    {
        // Default "Optional arguments" group
        parser->visible.add_argument("-t", "--threads")
                .help("Number of threads for processing. "
                      "Default uses all available threads.")
                .default_value(0)
                .scan<'i', int>();

        parser->visible.add_argument("--infer-threads")
                .help("Number of threads per device.")
                .default_value(1)
                .scan<'i', int>();

        cli::add_device_arg(*parser);

        parser->visible.add_argument("-v", "--verbose")
                .default_value(false)
                .implicit_value(true)
                .nargs(0)
                .action([&](const auto&) { ++verbosity; })
                .append();
    }
    {
        parser->visible.add_group("Input/output arguments");
        parser->visible.add_argument("-m", "--model-path").help("Path to correction model folder.");
    }
    {
        parser->visible.add_group("Advanced arguments");
        parser->visible.add_argument("-b", "--batch-size")
                .help("Batch size for inference. Default: 0 for auto batch size detection.")
                .default_value(128)
                .scan<'i', int>();
        parser->visible.add_argument("-w", "--window-size")
                .help("Window size for calling consensus.")
                .default_value(10000)
                .scan<'i', int>();
        parser->visible.add_argument("-w", "--window-overlap")
                .help("Overlap length between windows.")
                .default_value(1000)
                .scan<'i', int>();
    }

    return parser;
}

int parse_args(int argc, char** argv, utils::arg_parse::ArgParser& parser) {
    try {
        utils::arg_parse::parse(parser, argc, argv);

    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/// \brief This function simply fills out the Options struct with the parsed CLI args.
Options set_options(const utils::arg_parse::ArgParser& parser, const int verbosity) {
    Options opt;

    opt.in_aln_bam_fn = parser.visible.get<std::string>("in_aln_bam");
    opt.in_draft_fastx_fn = parser.visible.get<std::string>("in_draft_fastx");
    opt.out_consensus_fn = parser.visible.get<std::string>("out_consensus");

    opt.model_path = (parser.visible.is_used("--model-path"))
                             ? parser.visible.get<std::string>("model-path")
                             : "";

    opt.threads = parser.visible.get<int>("threads");
    opt.threads = (opt.threads == 0) ? std::thread::hardware_concurrency() : opt.threads;

    opt.infer_threads = parser.visible.get<int>("infer-threads");

    opt.device = parser.visible.get<std::string>("device");

    if (opt.device == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        opt.device = "cpu";
#else
        opt.device = utils::get_auto_detected_device();
#endif
    }

    opt.batch_size = parser.visible.get<int>("batch-size");
    opt.window_size = parser.visible.get<int>("window-size");
    opt.verbosity = verbosity;

    return opt;
}

void validate_options(const Options& opt) {
    // Parameter validation.
    if (!cli::validate_device_string(opt.device)) {
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_aln_bam_fn)) {
        spdlog::error("Input reads file {} does not exist!", opt.in_aln_bam_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_draft_fastx_fn)) {
        spdlog::error("Input reads file {} does not exist!", opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if ((opt.out_consensus_fn == opt.in_aln_bam_fn) ||
        (opt.out_consensus_fn == opt.in_draft_fastx_fn)) {
        spdlog::error("Output path matches one of the input paths!");
        std::exit(EXIT_FAILURE);
    }
    if (opt.batch_size < 0) {
        spdlog::error("Batch size should be > 0. Given: {}.", opt.batch_size);
        std::exit(EXIT_FAILURE);
    }
    if (opt.window_size <= 0) {
        spdlog::error("Window size should be > 0. Given: {}.", opt.window_size);
        std::exit(EXIT_FAILURE);
    }

    if (!std::empty(opt.model_path) && !std::filesystem::exists(opt.model_path)) {
        spdlog::error("Input model directory {} does not exist!", opt.model_path.string());
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

// void run_experimental_counts(const Options& opt) {
//     // Test-run the Medaka pileup code.
//     {
//         // const std::string_view region("contig_15:1-20001");
//         const std::string_view region("contig_15:1-5");

//         size_t num_dtypes = 1;
//         char **dtypes = NULL;
//         char tag_name[2] = "";
//         int tag_value = 0;
//         bool keep_missing = false;
//         size_t num_homop = 1;
//         bool weibull_summation = false;
//         const char* read_group = NULL;
//         const int min_mapQ = 1;

//         bam_fset* bam_set = create_bam_fset(opt.in_aln_bam_fn.c_str());

//         plp_data pileup = calculate_pileup(
//             region.data(), bam_set, num_dtypes, dtypes,
//             num_homop, tag_name, tag_value, keep_missing,
//             weibull_summation, read_group, min_mapQ);

//         print_pileup_data(pileup, num_dtypes, dtypes, num_homop);
//         fprintf(stdout, "pileup is length %zu, with buffer of %zu columns\n", pileup->n_cols, pileup->buffer_cols);
//         destroy_plp_data(pileup);
//         destroy_bam_fset(bam_set);
//     }
// }

void run_experimental(const Options& opt) {
    std::vector<std::string> devices;
    // int32_t infer_threads = 1;

    if (opt.device == "cpu") {
        // infer_threads = 1;
        devices.push_back(opt.device);
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(device, "cuda")) {
        devices = dorado::utils::parse_cuda_device_string(opt.device);
        if (devices.empty()) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + opt.device);
    }
#endif
    // const float batch_factor = (utils::starts_with(opt.device, "cuda")) ? 0.4f : 0.8f;
    // for (size_t d = 0; d < devices.size(); d++) {
    //     const auto& dev = devices[d];
    //     for (int i = 0; i < infer_threads; i++) {
    //         int device_batch_size = opt.batch_size;
    //         // if (batch_size == 0) {
    //         //     device_batch_size = calculate_batch_size(dev, batch_factor);
    //         //     if (device_batch_size == 0) {
    //         //         throw std::runtime_error("Insufficient memory to run inference on " + dev);
    //         //     }
    //         // }
    //         spdlog::info("Using batch size {} on device {} in inference thread {}.",
    //                      device_batch_size, dev, i);
    //         // m_infer_threads.push_back(std::thread(&CorrectionInferenceNode::infer_fn, this, dev,
    //         //                                       (int)d, device_batch_size));
    //     }
    // }

    const std::string device_str = devices.front();
    torch::Device device = torch::Device(device_str);

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (opt.device.is_cuda()) {
        stream = c10::cuda::getStreamFromPool(false, opt.device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    at::InferenceMode infer_guard;

    // const std::string model_path = (opt.model_path / "weights.pt").string(); // (m_model_config.model_dir / m_model_config.weights_file).string();
    const std::string model_path = opt.model_path / "model.pt";
    torch::jit::script::Module module;
    try {
        spdlog::debug("Loading model on {}...", device_str);
        module = torch::jit::load(model_path, device);
        spdlog::debug("Loaded model on {}!", device_str);
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path +
                                 " with error: " + e.what());
    }
    module.eval();

    {
        bam_fset* bam_set = create_bam_fset(opt.in_aln_bam_fn.c_str());
        const auto result = polisher::counts_feature_encoder(bam_set, "contig_15:1-5");
        std::cout << "result.feature_matrix =\n" << result.feature_matrix << "\n";
        std::cout << "result.positions =\n" << result.positions << "\n";
        destroy_bam_fset(bam_set);
    }
}

int polish(int argc, char* argv[]) {
    // Initialize CLI options. The parse_args below requires a non-const reference.
    // Verbosity is passed into a callback, so we need it here.
    int verbosity = 0;
    ParserPtr parser = create_cli(verbosity);

    // Parse the arguments.
    const int rv_parse = parse_args(argc, argv, *parser);

    if (rv_parse != EXIT_SUCCESS) {
        return rv_parse;
    }

    // Initialize the options from the CLI.
    const Options opt = set_options(*parser, verbosity);

    // Initialize the log level.
    if (opt.verbosity) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(opt.verbosity));
    }

    // Check if input options are good.
    validate_options(opt);

    if (std::empty(opt.model_path)) {
        throw std::runtime_error(
                "WIP. Currently can only load a model. Not yet fetching a model automatically.");
    }

    [[maybe_unused]] polisher::ModelConfig config =
            polisher::parse_model_config(opt.model_path / "config.toml");

    run_experimental(opt);

    return 0;
}

}  // namespace dorado
