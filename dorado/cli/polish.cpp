#include "cli/cli_utils.h"
#include "correct/infer.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_config.h"
#include "polish/architectures/model_factory.h"
#include "polish/medaka_bamiter.h"
#include "polish/medaka_counts.h"
#include "polish/polish_impl.h"
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

enum class OutputFormat {
    FASTA,
    FASTQ,
};

/// \brief All options for this tool.
struct Options {
    // Positional parameters.
    std::filesystem::path in_aln_bam_fn;
    std::filesystem::path in_draft_fastx_fn;

    // Optional parameters.
    std::filesystem::path out_consensus_fn;
    std::filesystem::path model_path;
    OutputFormat out_format = OutputFormat::FASTA;
    int32_t verbosity = 0;
    int32_t threads = 1;
    int32_t infer_threads = 1;
    std::string device_str;
    int32_t batch_size = 128;
    int64_t draft_batch_size = 200'000'000;
    int32_t window_len = 10000;
    int32_t window_overlap = 1000;
    int32_t bam_chunk = 1'000'000;
    std::string region;
    bool full_precision = false;
    int32_t min_mapq = 0;
    // int32_t min_depth = 0;
};

/// \brief Define the CLI options.
ParserPtr create_cli(int& verbosity) {
    ParserPtr parser = std::make_unique<utils::arg_parse::ArgParser>("dorado consensus");

    parser->visible.add_description("Consensus tool for polishing draft assemblies");

    {
        // Positional arguments group
        parser->visible.add_argument("in_aln_bam").help("Aligned reads in BAM format");
        parser->visible.add_argument("in_draft_fastx").help("Draft assembly for polishing");
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
        parser->visible.add_argument("-o", "--out-path")
                .help("Output to a file instead of stdout.");
        parser->visible.add_argument("-m", "--model-path").help("Path to correction model folder.");
        parser->visible.add_argument("-q", "--qualities")
                .help("Output with per-base quality scores (FASTQ).")
                .default_value(false)
                .implicit_value(true);
    }
    {
        parser->visible.add_group("Advanced arguments");
        parser->visible.add_argument("-b", "--batch-size")
                .help("Batch size for inference. Default: 0 for auto batch size detection.")
                .default_value(100)
                .scan<'i', int>();
        parser->visible.add_argument("--draft-batch-size")
                .help("Input draft sequences will be process in batches of roughly this size.")
                .default_value(std::string{"200M"});
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
        parser->visible.add_argument("--min-mapq")
                .help("Minimum mapping quality of alignment used for polishing.")
                .default_value(0)
                .scan<'i', int>();
        parser->visible.add_argument("--full-precision")
                .help("Always use full precision for inference.")
                .default_value(false)
                .implicit_value(true);

        // parser->visible.add_argument("--min-depth")
        //         .help("Sites with depth lower than min_depth will not be polished.")
        //         .default_value(0)
        //         .scan<'i', int>();
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

    opt.out_consensus_fn = (parser.visible.is_used("--out-path"))
                                   ? parser.visible.get<std::string>("out-path")
                                   : "";

    opt.model_path = (parser.visible.is_used("--model-path"))
                             ? parser.visible.get<std::string>("model-path")
                             : "";

    opt.out_format =
            parser.visible.get<bool>("qualities") ? OutputFormat::FASTQ : OutputFormat::FASTA;
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
    opt.draft_batch_size =
            std::max<int64_t>(0, utils::arg_parse::parse_string_to_size<int64_t>(
                                         parser.visible.get<std::string>("draft-batch-size")));
    opt.window_len = parser.visible.get<int>("window-len");
    opt.window_overlap = parser.visible.get<int>("window-overlap");
    opt.bam_chunk = parser.visible.get<int>("bam-chunk");
    opt.verbosity = verbosity;
    opt.region =
            (parser.visible.is_used("--region")) ? parser.visible.get<std::string>("region") : "";

    opt.full_precision = parser.visible.get<bool>("full-precision");
    opt.min_mapq = parser.visible.get<int>("min-mapq");
    // opt.min_depth = parser.visible.get<int>("min-depth");

    return opt;
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
    if (opt.batch_size <= 0) {
        spdlog::error("Batch size should be > 0. Given: {}.", opt.batch_size);
        std::exit(EXIT_FAILURE);
    }
    if (opt.draft_batch_size <= 0) {
        spdlog::error("Draft batch size should be > 0. Given: {}.", opt.draft_batch_size);
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

    if (!std::empty(opt.model_path) && !std::filesystem::exists(opt.model_path)) {
        spdlog::error("Input model directory {} does not exist!", opt.model_path.string());
        std::exit(EXIT_FAILURE);
    }

    if (!std::filesystem::exists(opt.in_aln_bam_fn) ||
        std::filesystem::is_empty(opt.in_aln_bam_fn)) {
        spdlog::error("Input file {} does not exist or is empty.", opt.in_aln_bam_fn.string());
        std::exit(EXIT_FAILURE);
    }
    if (!std::filesystem::exists(opt.in_draft_fastx_fn) ||
        std::filesystem::is_empty(opt.in_draft_fastx_fn)) {
        spdlog::error("Input file {} does not exist or is empty.", opt.in_draft_fastx_fn.string());
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
#endif
    else {
        throw std::runtime_error("Unsupported device: " + devices_str);
    }

    return devices;
}

std::unique_ptr<polisher::TorchModel> create_model(const std::filesystem::path& model_path,
                                                   const DeviceInfo& device_info,
                                                   const bool full_precision) {
    // Load weights from the model file.
    torch::jit::script::Module module;

    try {
        spdlog::debug("Loading weights from file: {}", model_path.string());
        module = torch::jit::load(model_path.string());
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading model from " + model_path.string() +
                                 " with error: " + e.what());
    }

    // Construct the model.
    spdlog::debug("Creating the GRU model.");
    std::unique_ptr<polisher::GRUModel> model =
            std::make_unique<polisher::GRUModel>(10, 5, 128, 2, true, true);

    spdlog::debug("Setting the weights.");
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
    model->to(device_info.device);
    if ((device_info.type == DeviceType::CUDA) && !full_precision) {
        model->to_half();
        spdlog::info("Converted the model to half.");
    } else {
        spdlog::info("Using full precision.");
    }
    model->eval();

    size_t total_params = 0;
    size_t total_bytes = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
        total_bytes += param.numel() * param.element_size();
    }
    spdlog::info("Model: total parameters: {}, size: {} MB", total_params,
                 (total_bytes / (1024.0 * 1024.0)));

    return model;
}

std::vector<std::pair<std::string, int64_t>> load_seq_lengths(
        const std::filesystem::path& in_fastx_fn) {
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

template <typename T, typename F>
std::vector<polisher::Interval> create_batches(const T& data,
                                               const int64_t batch_size,
                                               const F& functor_data_size) {
    std::vector<polisher::Interval> ret;
    polisher::Interval interval{0, 0};
    int64_t sum = 0;
    for (const auto& val : data) {
        const int64_t s = functor_data_size(val);
        sum += s;
        ++interval.end;
        if (sum >= batch_size) {
            ret.emplace_back(interval);
            interval.start = interval.end;
            sum = 0;
        }
    }
    if (interval.end > interval.start) {
        ret.emplace_back(interval);
    }
    return ret;
}

std::unique_ptr<std::ostream, void (*)(std::ostream*)> get_output_stream(
        const std::string& out_fn) {
    if (std::empty(out_fn)) {
        return {&std::cout, [](std::ostream*) {}};
    }
    std::unique_ptr<std::ofstream, void (*)(std::ostream*)> ofs(
            new std::ofstream(out_fn), [](std::ostream* ptr) { delete ptr; });
    if (!ofs->is_open()) {
        throw std::runtime_error("Failed to open file: " + out_fn);
    }
    return ofs;
}

}  // namespace

void run_polishing(const Options& opt, const std::vector<DeviceInfo>& devices) {
    if (std::empty(devices)) {
        spdlog::error("Zero devices initialized! Need at least one device to run.");
        std::exit(EXIT_FAILURE);
    }

    spdlog::info("Number of devices: {}", std::size(devices));

    at::InferenceMode infer_guard;

    const bool rv_fai = utils::create_fai_index(opt.in_draft_fastx_fn);
    if (!rv_fai) {
        spdlog::error("Failed to create/verify a .fai index for input file: '{}'!",
                      opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    spdlog::info("Loading draft sequence lengths.");
    const std::vector<std::pair<std::string, int64_t>> draft_lens =
            load_seq_lengths(opt.in_draft_fastx_fn);

    // Open the BAM file for each thread and spawn encoders.
    spdlog::info("Creating {} encoders.", opt.threads);
    std::vector<bam_fset*> bam_sets;
    std::vector<polisher::CountsFeatureEncoder> encoders;
    for (int32_t i = 0; i < opt.threads; ++i) {
        bam_sets.emplace_back(create_bam_fset(opt.in_aln_bam_fn.c_str()));
        encoders.emplace_back(polisher::CountsFeatureEncoder(bam_sets.back(), opt.min_mapq));
    }

    at::set_num_interop_threads(opt.threads);
    torch::set_num_threads(1);

    // Construct the model.
    spdlog::info("Loading the model.");
    const auto create_models = [&]() {
        std::vector<std::shared_ptr<polisher::TorchModel>> ret;
        for (int32_t device_id = 0; device_id < dorado::ssize(devices); ++device_id) {
            ret.emplace_back(create_model(opt.model_path / "model.pt", devices[device_id],
                                          opt.full_precision));
            spdlog::info("Loaded model to device {}: {}", device_id, devices[device_id].name);
        }
        if ((std::size(devices) == 1) && (devices.front().type == DeviceType::CPU)) {
            for (int32_t i = 1; i < opt.threads; ++i) {
                ret.emplace_back(ret.front());
            }
        }
        return ret;
    };
    const std::vector<std::shared_ptr<polisher::TorchModel>> models = create_models();

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    const std::vector<polisher::Interval> draft_batches =
            create_batches(draft_lens, opt.draft_batch_size,
                           [](const std::pair<std::string, int64_t>& val) { return val.second; });

    // Open the output stream, to std::cout if the path is empty, otherwise to the file.
    auto ofs = get_output_stream(opt.out_consensus_fn);

    for (const auto& draft_batch : draft_batches) {
        spdlog::info("=============================");
        spdlog::info("Starting to produce consensus for draft sequences: {}-{}/{}.",
                     draft_batch.start, draft_batch.end, std::size(draft_lens));

        spdlog::debug("Creating BAM windows.");
        const std::vector<std::pair<std::string, int64_t>> draft_lens_batch(
                std::begin(draft_lens) + draft_batch.start,
                std::begin(draft_lens) + draft_batch.end);
        const std::vector<polisher::Window> bam_regions = polisher::create_bam_regions(
                draft_lens_batch, opt.bam_chunk, opt.window_overlap, opt.region);

        // std::vector<std::vector<polisher::ConsensusResult>> pieces(std::size(draft_lens_batch));

        auto [samples, trims] =
                polisher::create_samples(encoders, bam_regions, draft_lens_batch, opt.threads,
                                         opt.window_len, opt.window_overlap);

        spdlog::info("Produced num samples: {}", std::size(samples));

        const polisher::CountsFeatureDecoder decoder(polisher::LabelSchemeType::HAPLOID);
        std::vector<polisher::ConsensusResult> results_samples =
                polisher::process_samples_in_parallel(samples, trims, models, decoder,
                                                      opt.window_len, opt.batch_size);

        // Group samples by sequence ID.
        std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(std::size(draft_lens_batch));
        for (int32_t i = 0; i < dorado::ssize(results_samples); ++i) {
            const polisher::ConsensusResult& r = results_samples[i];
            groups[r.draft_id].emplace_back(r.draft_start, i);
        }
        // Stitch the windows.
        for (size_t seq_id = 0; seq_id < std::size(groups); ++seq_id) {
            auto& group = groups[seq_id];

            // Sort by region start position, for every sequence.
            std::sort(std::begin(group), std::end(group));

            const polisher::ConsensusResult consensus =
                    polisher::stitch_sequence(opt.in_draft_fastx_fn, draft_lens_batch[seq_id].first,
                                              results_samples, group, seq_id);

            const std::string& header = draft_lens_batch[seq_id].first;
            write_consensus_result(*ofs, header, consensus,
                                   (opt.out_format == OutputFormat::FASTQ));
        }
    }

    for (auto& bam_set : bam_sets) {
        destroy_bam_fset(bam_set);
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
