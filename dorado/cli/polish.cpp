#include "cli/cli_utils.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/features.h"
#include "polish/medaka_bamiter.h"
#include "polish/medaka_counts.h"
#include "polish/model.h"
#include "polish/sample.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
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
    int32_t window_len = 10000;
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
        parser->visible.add_argument("out_consensus").help("Output consensus FASTA/FASTQ file.");
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
        parser->visible.add_argument("-w", "--window-len")
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
    opt.window_len = parser.visible.get<int>("window-len");
    opt.window_overlap = parser.visible.get<int>("window-overlap");
    opt.verbosity = verbosity;

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
    if (opt.window_len <= 0) {
        spdlog::error("Window size should be > 0. Given: {}.", opt.window_len);
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

// void print_output(const c10::IValue& output) {
//     if (output.isTensor()) {
//         // Single tensor output
//         std::cout << "Single tensor output: " << output.toTensor() << "\n";

//         // const torch::Tensor row_sum = output.toTensor().sum(1);
//         // std::cout << "Sum: " << row_sum << "\n";

//     } else if (output.isTuple()) {
//         // Tuple output - unpack and print each element
//         auto outputTuple = output.toTuple()->elements();
//         std::cout << "Tuple output with " << outputTuple.size() << " elements:" << std::endl;
//         for (size_t i = 0; i < outputTuple.size(); ++i) {
//             if (outputTuple[i].isTensor()) {
//                 std::cout << "Element " << i << ":\n";
//                 std::cout << outputTuple[i].toTensor() << "\n";
//             } else {
//                 std::cout << "Element " << i << " is not a tensor." << std::endl;
//             }
//         }
//     } else {
//         std::cout << "Output is neither a tensor nor a tuple." << std::endl;
//     }
// }

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

struct Window {
    std::string name;
    int64_t start = 0;
    int64_t end = 0;
    int64_t length = 0;
    int32_t num_windows = 0;
};

std::vector<Window> create_windows(const std::vector<std::pair<std::string, int64_t>>& seq_lens,
                                   const int32_t window_len,
                                   const int32_t window_overlap) {
    if (window_overlap >= window_len) {
        spdlog::error(
                "The window overlap cannot be larger than the window size! window_len = {}, "
                "window_overlap = {}\n",
                window_len, window_overlap);
        return {};
    }
    std::vector<Window> ret;
    for (int32_t seq_id = 0; seq_id < static_cast<int32_t>(std::size(seq_lens)); ++seq_id) {
        const auto& [name, length] = seq_lens[seq_id];
        const int32_t num_windows =
                static_cast<int32_t>(std::ceil(static_cast<double>(length) / window_len));
        ret.reserve(std::size(ret) + num_windows);
        for (int64_t start = 0; start < length; start += (window_len - window_overlap)) {
            const int64_t end = std::min(length, start + window_len);
            ret.emplace_back(Window{name, start, end, length, num_windows});
            if (end == length) {
                break;
            }
        }
    }
    return ret;
}

void run_experimental(const Options& opt) {
    std::vector<std::string> devices;
    // int32_t infer_threads = 1;

    // Check the output extension to determine if we need
    // to compute the QVs too.
    const std::string ext = get_lowercase_extension(opt.out_consensus_fn);
    const bool with_quals = ((ext == ".fastq") && (ext != ".fq")) ? true : false;

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

    std::array<std::mutex, 32> gpu_mutexes;  // One per GPU.

    auto batch_infer = [&gpu_mutexes, &device, &module](std::vector<polisher::Sample>& samples,
                                                        const int32_t mtx_idx) {
        utils::ScopedProfileRange infer("infer", 1);

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        std::vector<torch::Tensor> batch_features;
        for (auto& sample : samples) {
            batch_features.emplace_back(std::move(sample.features));
        }
        torch::Tensor batch_features_tensor = torch::stack(batch_features);

        std::unique_lock<std::mutex> lock(gpu_mutexes[mtx_idx]);
        std::vector<torch::jit::IValue> inputs;
        {
            utils::ScopedProfileRange move_to_device("move_to_device", 1);
            inputs.push_back(batch_features_tensor.to(device));
        }

        c10::IValue output;
        try {
            output = module.forward(inputs);
        } catch (std::runtime_error& e) {
#if DORADO_CUDA_BUILD
            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.what());
            c10::cuda::CUDACachingAllocator::emptyCache();
            output = module.forward(inputs);
#else
            throw e;
#endif
        }
        lock.unlock();

        spdlog::info("Inference done.");

        return output.toTensor();
    };

    {
        bam_fset* bam_set = create_bam_fset(opt.in_aln_bam_fn.c_str());

        spdlog::info("Loading draft sequence lengths.");

        const std::vector<std::pair<std::string, int64_t>> draft_lens =
                load_seq_lengths(opt.in_draft_fastx_fn);

        // for (size_t i = 0; i < std::size(draft_lens); ++i) {
        //     const auto& [name, length] = draft_lens[i];
        //     std::cerr << "[draft i = " << i << "] name = " << name << ", length = " << length << "\n";
        // }

        spdlog::info("Creating windows.");

        const std::vector<Window> windows =
                create_windows(draft_lens, opt.window_len, opt.window_overlap);

        spdlog::info("Created {} windows from {} sequences.", std::size(windows),
                     std::size(draft_lens));

        // for (size_t i = 0; i < std::size(windows); ++i) {
        //     std::cerr << "[window i = " << i << "] name = " << windows[i].name << ", start = " << windows[i].start << ", end = " << windows[i].end << ", len = " << windows[i].length << ", num_windows = " << windows[i].num_windows << "\n";
        // }

        std::ofstream ofs(opt.out_consensus_fn);

        for (const auto& window : windows) {
            // const std::string region_name = "contig_15";
            // const int64_t region_start = 0;
            // const int64_t region_end = 10000;

            polisher::CountsFeatureEncoder encoder(bam_set);
            std::vector<polisher::Sample> samples =
                    encoder.encode_region(window.name, window.start, window.end);

            // for (size_t i = 0; i < std::size(samples); ++i) {
            //     std::cerr << "[i = " << i << "] samples[i].features =\n"
            //               << samples[i].features << "\n";
            //     std::cerr << "[i = " << i << "] samples[i].positions =\n"
            //               << samples[i].positions << "\n";
            // }

            std::vector<polisher::Sample> samples_to_infer;
            std::vector<polisher::Sample> remainders;
            for (auto& sample : samples) {
                // if (static_cast<int32_t>(std::size(sample.positions)) < opt.window_len) {
                //     remainders.emplace_back(std::move(sample));
                //     continue;
                // }
                samples_to_infer.emplace_back(std::move(sample));
            }

            spdlog::info("Inference on batch size: {}", std::size(samples_to_infer));
            spdlog::info("Remainders size: {}", remainders.size());

            const torch::Tensor output = batch_infer(samples_to_infer, 0);

            std::vector<polisher::ConsensusResult> results =
                    encoder.decode_bases(output, with_quals);

            // Write output.
            {
                for (size_t i = 0; i < std::size(results); ++i) {
                    std::string& seq = results[i].seq;
                    std::string& quals = results[i].quals;

                    size_t n = 0;
                    for (size_t j = 0; j < std::size(seq); ++j) {
                        if (seq[j] == '*') {
                            continue;
                        }
                        seq[n] = seq[j];
                        quals[n] = quals[j];
                        ++n;
                    }
                    seq.resize(n);
                    quals.resize(n);
                    // seq.erase(std::remove(seq.begin(), seq.end(), '*'), seq.end());

                    if (with_quals) {
                        ofs << "@consensus-" << window.name << ':' << (window.start + 1) << '-'
                            << window.end << '\n'
                            << results[i].seq << "\n+\n"
                            << results[i].quals << '\n';
                    } else {
                        ofs << ">consensus-" << window.name << ':' << (window.start + 1) << '-'
                            << window.end << '\n'
                            << seq << '\n';
                    }
                }
            }

            // const at::Tensor collated_features = collate<float>(quals_batch, 0.f, torch::kFloat32);

            // TODO here: Current code implements the logic in SampleGenerator._fill_features and everything upstream.
            //              But, I still need to implement SampleGenerator.samples() which separates the small chunks into "quarantined" samples,
            //              and also splits large chunks of the Sample object into smaller overlapping ones.
            //              This is implemented in the medaka.common.Sample.chunks() function.
            // Workflow is:
            //  SampleGenerator.samples() <-prediction.DataLoader._run_region() <- prediction.DataLoader._region_worker() <- DataLoader.__init__()
            //
            // Note that there is also a path SampleGenerator.samples() <- SampleGenerator._samples_worker <- SampleGenerator.create_samples, but that one is called
            //  from a different subtool, `features`.
            //
            // TODO 2: All windows which are of length chunk_size have exactly the same dimensions (since the rows are fixed to counts and not actual reads), so no padding
            //          is needed. Instead, we can just stack these features into this batch tensor.
            //          SHORT windows need special handling. In Medaka, these are interchangeably called "_quarantined" or "remainders" (self.remainders.extend(remain)).
            //          Medaka currently calls these windows with batch_size = 1, which means it does not do any sort of padding, but instead it only runs individual
            //          windows, which may be wasteful.
            //          We could pad everything to window_len length and just push to the same tensor for batch processing. Potentially use Joyjit's function
            //          from Dorado Correct to `collate`.
        }

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
