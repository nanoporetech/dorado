#include "cli/cli_utils.h"
#include "correct/infer.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/features.h"
#include "polish/medaka_bamiter.h"
#include "polish/medaka_counts.h"
#include "polish/model.h"
#include "polish/polish_impl.h"
#include "polish/polish_models.h"
#include "polish/sample.h"
#include "polish/trim.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/ssize.h"

#include <cxxpool.h>
#include <htslib/faidx.h>
#include <spdlog/spdlog.h>
#include <torch/script.h>
#include <torch/torch.h>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif

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
#include <tuple>
#include <unordered_set>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

// #define DEBUG_POLISH_DUMP_SEQ_PIECES
// #define DEBUG_POLISH_REGIONS
// #define DEBUG_POLISH_SAMPLE_CONSTRUCTION

namespace dorado {

namespace {

using ParserPtr = std::unique_ptr<utils::arg_parse::ArgParser>;

enum class DeviceType { CPU, CUDA, METAL, UNKNOWN };

struct DeviceInfo {
    std::string name;
    DeviceType type;
    torch::Device device;
};

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
    std::string device_str;
    int32_t batch_size = 128;
    int32_t window_len = 10000;
    int32_t window_overlap = 1000;
    int32_t bam_chunk = 1000000;
    std::string region;
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado consensus");

    parser->visible.add_description("Consensus tool for polishing draft assemblies");

    {
        // Positional arguments group
        parser->visible.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser->visible.add_argument("in_draft_fastx").help("Draft assembly for polishing");
        parser->visible.add_argument("out_consensus").help("Output consensus FASTA/FASTQ file.");
    }
    {
        // Default "Optional arguments" group
        parser->visible.add_argument("-t", "--threads")
                .help("Number of threads for processing. "
                      "Default uses all available threads.")
                .default_value(1)
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
                .default_value(100)
                .scan<'i', int>();
        parser->visible.add_argument("-w", "--window-len")
                .help("Window size for calling consensus.")
                .default_value(10000)
                .scan<'i', int>();
        parser->visible.add_argument("-w", "--window-overlap")
                .help("Overlap length between windows.")
                .default_value(1000)
                .scan<'i', int>();
        parser->visible.add_argument("--bam-chunk")
                .help("Size of draft chunks to parse from the input BAM at a time.")
                .default_value(1000000)
                .scan<'i', int>();
        parser->visible.add_argument("--region")
                .help("Process only this region of the input. Htslib format (start is 1-based, end "
                      "is inclusive).");
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

    opt.device_str = parser.visible.get<std::string>("device");

    if (opt.device_str == cli::AUTO_DETECT_DEVICE) {
#if DORADO_METAL_BUILD
        opt.device_str = "cpu";
#else
        opt.device_str = utils::get_auto_detected_device();
#endif
    }

    opt.batch_size = parser.visible.get<int>("batch-size");
    opt.window_len = parser.visible.get<int>("window-len");
    opt.window_overlap = parser.visible.get<int>("window-overlap");
    opt.bam_chunk = parser.visible.get<int>("bam-chunk");
    opt.verbosity = verbosity;
    opt.region =
            (parser.visible.is_used("--region")) ? parser.visible.get<std::string>("region") : "";
    return opt;
}

std::string get_lowercase_extension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(std::begin(ext), std::end(ext), std::begin(ext),
                   [](unsigned char c) { return std::tolower(c); });
    return ext;
}

void validate_options(const Options& opt) {
    // Parameter validation.
    if (!cli::validate_device_string(opt.device_str)) {
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_aln_bam_fn)) {
        spdlog::error("Input draft file {} does not exist!", opt.in_aln_bam_fn.string());
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
    if (opt.window_len <= 0) {
        spdlog::error("Window size should be > 0. Given: {}.", opt.window_len);
        std::exit(EXIT_FAILURE);
    }
    if (opt.bam_chunk <= 0) {
        spdlog::error("BAM chunk size should be > 0. Given: {}.", opt.bam_chunk);
        std::exit(EXIT_FAILURE);
    }
    if ((opt.window_overlap < 0) || (opt.window_overlap >= opt.window_len)) {
        spdlog::error(
                "Window overlap should be >= 0 and < window_len. Given: window_overlap = {}, "
                "window_len = {}.",
                opt.window_overlap, opt.window_len);
        std::exit(EXIT_FAILURE);
    }
    {
        const std::string ext = get_lowercase_extension(opt.out_consensus_fn);
        if ((ext != ".fasta") && (ext != ".fastq") && (ext != ".fa") && (ext != ".fq")) {
            spdlog::error(
                    "Unknown extension of output file: {}. Supported: .fasta, .fastq, .fa, .fq.",
                    opt.out_consensus_fn.string());
            std::exit(EXIT_FAILURE);
        }
    }

    if (!std::empty(opt.model_path) && !std::filesystem::exists(opt.model_path)) {
        spdlog::error("Input model directory {} does not exist!", opt.model_path.string());
        std::exit(EXIT_FAILURE);
    }
}

std::vector<DeviceInfo> init_devices(const std::string& devices_str) {
    std::vector<DeviceInfo> devices;

    if (devices_str == "cpu") {
        torch::Device torch_device = torch::Device(devices_str);
        devices.emplace_back(DeviceInfo{devices_str, DeviceType::CPU, std::move(torch_device)});
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(devices_str, "cuda")) {
        spdlog::info("Parsing CUDA device string.");
        const std::vector<std::string> parsed_devices =
                dorado::utils::parse_cuda_device_string(devices_str);
        if (parsed_devices.empty()) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
        for (const auto& val : parsed_devices) {
            torch::Device torch_device = torch::Device(val);
            devices.emplace_back(DeviceInfo{val, DeviceType::CUDA, std::move(torch_device)});
        }
    }
#else
    else {
        throw std::runtime_error("Unsupported device: " + devices_str);
    }
#endif

    return devices;
}

std::unique_ptr<polisher::TorchModel> create_model(const std::filesystem::path& model_path,
                                                   const DeviceInfo& device_info) {
    // Load weights from the model file.
    torch::jit::script::Module module;

    try {
        spdlog::info("Loading weights from file: {}", model_path.string());
        module = torch::jit::load(model_path.string());
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path.string() +
                                 " with error: " + e.what());
    }

    // Construct the model.
    spdlog::info("Creating the GRU model.");
    std::unique_ptr<polisher::GRUModel> model = std::make_unique<polisher::GRUModel>(10, 5, 128);

    spdlog::info("Setting the weights.");
    auto state_dict = module.named_parameters();
    for (const auto& p : state_dict) {
        auto* param = model->named_parameters().find(p.name);
        if (param != nullptr) {
            param->copy_(p.value);
        } else {
            throw std::runtime_error(
                    "Some loaded parameters cannot be found in the C++ model! name = " + p.name);
        }
    }
    spdlog::info("Moving the model to the device: {}", device_info.name);
    model->to(device_info.device);
    spdlog::info("Moved the model to the device: {}. Converting to half.", device_info.name);
    if (device_info.type == DeviceType::CUDA) {
        model->to_half();
        spdlog::info("Converted the model to half.");
    }
    spdlog::info("Calling model->eval().");
    model->eval();
    spdlog::info("Model ready.");

    size_t total_params = 0;
    size_t total_bytes = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
        total_bytes += param.numel() * param.element_size();
    }
    spdlog::info("Total parameters: {}", total_params);
    spdlog::info("Total size (in MB): {} MB", (total_bytes / (1024.0 * 1024.0)));

    return model;
}

std::vector<std::pair<std::string, int64_t>> load_seq_lengths(
        const std::filesystem::path& in_fastx_fn) {
    if (!utils::check_fai_exists(in_fastx_fn)) {
        utils::create_fai_index(in_fastx_fn);
    }

    const std::filesystem::path fai_path = utils::get_fai_path(in_fastx_fn);

    std::vector<std::pair<std::string, int64_t>> ret;
    std::string line;
    std::ifstream ifs(fai_path);
    while (std::getline(ifs, line)) {
        if (std::empty(line)) {
            continue;
        }
        std::string name;
        int64_t length = 0;
        std::istringstream iss(line);
        iss >> name >> length;
        ret.emplace_back(std::move(name), length);
    }
    return ret;
}

void write_consensus_result(std::ostream& os,
                            const std::string& seq_name,
                            const polisher::ConsensusResult& result,
                            const bool write_quals) {
    if (std::empty(result.seq)) {
        return;
    }

    polisher::ConsensusResult out = result;
    remove_deletions(out);

    if (write_quals) {
        os << '@' << seq_name << '\n' << out.seq << "\n+\n" << out.quals << '\n';
    } else {
        os << '>' << seq_name << '\n' << out.seq << '\n';
    }
}

}  // namespace

void run_polishing(const Options& opt, const std::vector<DeviceInfo>& devices) {
    if (std::empty(devices)) {
        spdlog::error("Zero devices initialized! Need at least one device to run.");
        std::exit(EXIT_FAILURE);
    }

    // Check the output extension to determine if we need to compute the QVs too.
    const std::string ext = get_lowercase_extension(opt.out_consensus_fn);
    const bool with_quals = ((ext == ".fastq") && (ext != ".fq")) ? true : false;

    spdlog::info("Number of devices: {}", std::size(devices));

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (device.is_cuda()) {
        spdlog::info("Acquiring a CUDA stream.");
        stream = c10::cuda::getStreamFromPool(false, device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    at::InferenceMode infer_guard;

    // Main processing code.
    {
        spdlog::info("Loading draft sequence lengths.");

        const std::vector<std::pair<std::string, int64_t>> draft_lens =
                load_seq_lengths(opt.in_draft_fastx_fn);

        // IMPORTANT: The intra-thread parallelism was killing performance when multiple threads were used.
        //              Remember to reset this at the inference stage so that the CPU-only runs don't suffer.
        at::set_num_interop_threads(opt.threads);
        torch::set_num_threads(1);

        // Create BAM windows (regions) to create pileup. The features (samples) will
        // be split further into windows of window_len in size prior to inference.
        spdlog::info("Creating BAM windows.");
        const std::vector<polisher::Window> bam_regions = polisher::create_bam_regions(
                draft_lens, opt.bam_chunk, opt.window_overlap, opt.region);

        // Encode samples (features) in parallel. A window can have multiple samples if there was a gap.
        auto [samples, trims] =
                polisher::create_samples(opt.in_aln_bam_fn, bam_regions, draft_lens, opt.threads,
                                         opt.window_len, opt.window_overlap);

        spdlog::info("Num samples: {}", samples.size());

        // Increase the number of threads again for inter-op parallelism.
        torch::set_num_threads(opt.threads);

        // Construct the model.
        spdlog::info("Loading the model.");
        const std::vector<std::shared_ptr<polisher::TorchModel>> models = [&]() {
            std::vector<std::shared_ptr<polisher::TorchModel>> ret;
            for (int32_t device_id = 0; device_id < dorado::ssize(devices); ++device_id) {
                ret.emplace_back(create_model(opt.model_path / "model.pt", devices[device_id]));
                spdlog::info("Loaded model to device {}: {}", device_id, devices[device_id].name);
                std::cerr << "devices[device_id].device = " << devices[device_id].device << "\n";
            }
            if ((std::size(devices) == 1) && (devices.front().type == DeviceType::CPU)) {
                for (int32_t i = 1; i < opt.threads; ++i) {
                    ret.emplace_back(models.front());
                }
            }
            return ret;
        }();

        spdlog::info("Processing samples in batches. Num samples: {}.", std::size(samples));

        const polisher::CountsFeatureDecoder decoder;
        std::vector<polisher::ConsensusResult> results_samples =
                polisher::process_samples_in_parallel(samples, models, decoder, opt.window_len,
                                                      opt.batch_size);

        // Stitching information, collect all samples for each sequence.
        std::vector<std::vector<std::pair<int64_t, int32_t>>> samples_for_seqs(
                std::size(draft_lens));
        for (int32_t i = 0; i < static_cast<int32_t>(std::size(samples)); ++i) {
            const polisher::Sample& sample = samples[i];
            samples_for_seqs[sample.seq_id].emplace_back(sample.start(), i);
        }

        std::ofstream ofs(opt.out_consensus_fn);

        // Stitch the windows.
        for (size_t seq_id = 0; seq_id < std::size(samples_for_seqs); ++seq_id) {
            auto& samples_for_seq = samples_for_seqs[seq_id];

            // Sort by region start position, for every sequence.
            std::sort(std::begin(samples_for_seq), std::end(samples_for_seq));

            if (std::empty(samples_for_seq)) {
                spdlog::warn("Sequence {} has zero inferred windows.", draft_lens[seq_id].first);
                continue;
            }

            std::vector<int32_t> sample_ids;
            sample_ids.reserve(std::size(samples_for_seq));
            for (const auto& [_, sample_id] : samples_for_seq) {
                sample_ids.emplace_back(sample_id);
            }

            const polisher::ConsensusResult consensus = polisher::stitch_sequence(
                    opt.in_draft_fastx_fn, draft_lens[seq_id].first, samples, trims,
                    results_samples, samples_for_seq, seq_id);

            const std::string header = std::string("consensus") + "-" + draft_lens[seq_id].first;
            write_consensus_result(ofs, header, consensus, with_quals);
        }
    }

    spdlog::info("Done!");
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

    spdlog::set_level(spdlog::level::info);

    // Initialize the log level.
    if (opt.verbosity) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(opt.verbosity));
    }

    spdlog::flush_every(std::chrono::seconds(1));

    // Check if input options are good.
    validate_options(opt);

    if (std::empty(opt.model_path)) {
        throw std::runtime_error(
                "WIP. Currently can only load a model. Not yet fetching a model automatically.");
    }

    [[maybe_unused]] polisher::ModelConfig config =
            polisher::parse_model_config(opt.model_path / "config.toml");

    const std::vector<DeviceInfo> devices = init_devices(opt.device_str);

    if (std::empty(devices)) {
        throw std::runtime_error("Zero devices initialized! Need at least one device to run.");
    }

    run_polishing(opt, devices);

    return 0;
}

}  // namespace dorado
