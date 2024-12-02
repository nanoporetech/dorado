#include "cli/cli_utils.h"
#include "correct/infer.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/architectures/decoder_factory.h"
#include "polish/architectures/encoder_factory.h"
#include "polish/architectures/model_config.h"
#include "polish/architectures/model_factory.h"
#include "polish/bam_file.h"
#include "polish/interval.h"
#include "polish/medaka_counts.h"
#include "polish/polish_impl.h"
#include "polish/sample.h"
#include "polish/trim.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/AsyncQueue.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/ssize.h"
#include "utils/timer_high_res.h"

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
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
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

struct PolisherResources {
    std::unique_ptr<polisher::EncoderBase> encoder;
    std::unique_ptr<polisher::FeatureDecoder> decoder;
    std::vector<BamFile> bam_handles;
    std::vector<DeviceInfo> devices;
    std::vector<std::shared_ptr<polisher::ModelTorchBase>> models;
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
    int32_t threads = 0;
    int32_t infer_threads = 1;
    bool infer_threads_is_set = false;
    std::string device_str;
    int32_t batch_size = 128;
    int64_t draft_batch_size = 200'000'000;
    int32_t window_len = 10000;
    int32_t window_overlap = 1000;
    int32_t bam_chunk = 1'000'000;
    int32_t bam_subchunk = 100'000;
    std::string region;
    bool full_precision = false;
    bool load_scripted_model = false;
    int32_t num_preloaded_batches = 1000;
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
        parser->visible.add_argument("--bam-subchunk")
                .help("Each BAM region of bam_chunk length will be split into non-overlapping "
                      "regions of this size for parallel processing.")
                .default_value(100000)
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
        parser->visible.add_argument("--scripted")
                .help("Load the scripted Torch model instead of building one internally.")
                .default_value(false)
                .implicit_value(true);
        parser->visible.add_argument("--preloaded-batches")
                .help("Maximum number of preloaded batches for inference.")
                .default_value(1000)
                .scan<'i', int>();

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
    opt.threads = (opt.threads == 0) ? std::max(1U, std::thread::hardware_concurrency() / 2)
                                     : opt.threads;

    opt.infer_threads = parser.visible.get<int>("infer-threads");
    opt.infer_threads_is_set = parser.visible.is_used("--infer-threads");

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
    opt.bam_subchunk = parser.visible.get<int>("bam-subchunk");
    opt.verbosity = verbosity;
    opt.region =
            (parser.visible.is_used("--region")) ? parser.visible.get<std::string>("region") : "";

    opt.full_precision = parser.visible.get<bool>("full-precision");
    opt.load_scripted_model = parser.visible.get<bool>("scripted");
    // opt.min_depth = parser.visible.get<int>("min-depth");

    opt.num_preloaded_batches = parser.visible.get<int>("preloaded-batches");

    if (opt.bam_subchunk > opt.bam_chunk) {
        spdlog::warn(
                "BAM sub-chunk size is larger than bam_chunk size. Limiting to bam_chunk size. "
                "bam_subchunk = {}, bam_chunk = {}",
                opt.bam_chunk, opt.bam_subchunk);
        opt.bam_subchunk = opt.bam_chunk;
    }

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
    if (opt.bam_subchunk <= 0) {
        spdlog::error("BAM sub-chunk size should be > 0. Given: {}.", opt.bam_chunk);
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

    if ((opt.device_str != "cpu") && opt.infer_threads_is_set) {
        spdlog::error(
                "Specifying the number of CPU inference threads is only allowed when the device is "
                "set to 'cpu'.");
        std::exit(EXIT_FAILURE);
    }

    if (opt.num_preloaded_batches <= 0) {
        spdlog::error("Number of preloaded batches needs to be > 0, given: {}.",
                      opt.num_preloaded_batches);
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

[[maybe_unused]] void write_consensus_result(std::ostream& os,
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

std::unique_ptr<std::ostream, void (*)(std::ostream*)> get_output_stream(
        const std::filesystem::path& out_fn) {
    if (std::empty(out_fn)) {
        return {&std::cout, [](std::ostream*) {}};
    }
    std::unique_ptr<std::ofstream, void (*)(std::ostream*)> ofs(
            new std::ofstream(out_fn), [](std::ostream* ptr) { delete ptr; });
    if (!ofs->is_open()) {
        throw std::runtime_error("Failed to open file: " + out_fn.string());
    }
    return ofs;
}

PolisherResources create_resources(const polisher::ModelConfig& model_config,
                                   const std::filesystem::path& in_aln_bam_fn,
                                   const std::string& device_str,
                                   const int32_t num_bam_threads,
                                   const int32_t num_inference_cpu_threads,
                                   const bool full_precision) {
    PolisherResources resources;

    spdlog::info("Initializing the devices.");
    resources.devices = init_devices(device_str);
    if (std::empty(resources.devices)) {
        throw std::runtime_error("Zero devices initialized! Need at least one device to run.");
    }

    // Construct the model.
    spdlog::info("Loading the model.");
    const auto create_models = [&]() {
        std::vector<std::shared_ptr<polisher::ModelTorchBase>> ret;

        for (int32_t device_id = 0; device_id < dorado::ssize(resources.devices); ++device_id) {
            const auto& device_info = resources.devices[device_id];

            spdlog::info("Creating a model from the config.");
            auto model = polisher::model_factory(model_config);

            spdlog::info("About to load model to device {}: {}", device_id, device_info.name);
            model->to_device(device_info.device);

            // Half-precision if needed.
            if ((device_info.type == DeviceType::CUDA) && !full_precision) {
                spdlog::info("Converting the model to half.");
                model->to_half();
            } else {
                spdlog::info("Using full precision.");
            }

            spdlog::info("Switching model to eval mode.");
            model->set_eval();

            ret.emplace_back(std::move(model));

            spdlog::info("Loaded model to device {}: {}", device_id, device_info.name);
        }
        // In case the device is set to CPU, we add up to inference_threads models.
        if ((std::size(resources.devices) == 1) &&
            (resources.devices.front().type == DeviceType::CPU)) {
            for (int32_t i = 1; i < num_inference_cpu_threads; ++i) {
                ret.emplace_back(ret.front());
            }
        }
        return ret;
    };
    resources.models = create_models();

    spdlog::info("Creating the encoder.");
    resources.encoder = polisher::encoder_factory(model_config);

    spdlog::info("Creating the decoder.");
    resources.decoder = polisher::decoder_factory(model_config);

    // Open the BAM file for each thread.
    spdlog::info("Creating {} BAM handles.", num_bam_threads);
    for (int32_t i = 0; i < num_bam_threads; ++i) {
        resources.bam_handles.emplace_back(BamFile(in_aln_bam_fn));
    }

    return resources;
}

}  // namespace

namespace polisher {

void sample_producer(PolisherResources& resources,
                     const std::vector<polisher::Window>& bam_regions,
                     const std::vector<std::pair<std::string, int64_t>>& draft_lens,
                     const int32_t num_threads,
                     const int32_t batch_size,
                     const int32_t window_len,
                     const int32_t window_overlap,
                     const int32_t bam_subchunk_len,
                     utils::AsyncQueue<InferenceData>& infer_data) {
    // (void)resources;
    // (void)bam_regions;
    // (void)draft_lens;
    // (void)num_threads;
    // (void)window_len;
    // (void)window_overlap;
    // (void)bam_subchunk_len;

    // const int32_t buffer_size = dorado::ssize(resources.devices) * batch_size;
    // const int32_t approx_max_buffer_size = buffer_size * 2;

    spdlog::debug("[producer] Input: {} BAM windows from {} sequences.", std::size(bam_regions),
                  std::size(draft_lens));

    // Split large BAM regions into non-overlapping windows for parallel encoding.
    // The non-overlapping windows will be merged after samples are constructed.
    std::vector<Window> windows;
    std::vector<Interval> bam_region_intervals;
    for (int32_t i = 0; i < static_cast<int32_t>(std::size(bam_regions)); ++i) {
        const Window& bw = bam_regions[i];
        std::vector<Window> new_windows =
                create_windows(bw.seq_id, bw.start, bw.end, bw.seq_length, bam_subchunk_len, 0, i);
        if (std::empty(new_windows)) {
            bam_region_intervals.emplace_back(Interval{0, 0});
            continue;
        }
        const int32_t num_windows = static_cast<int32_t>(std::size(windows));
        const int32_t num_new_windows = static_cast<int32_t>(std::size(new_windows));
        bam_region_intervals.emplace_back(Interval{num_windows, num_windows + num_new_windows});
        windows.reserve(std::size(windows) + std::size(new_windows));
        windows.insert(std::end(windows), std::begin(new_windows), std::end(new_windows));
    }

    // // A BAM region needs to be processed fully and then split into overlapping samples for inference.
    // // We cannot predict how many samples there will be in a BAM region due to indels, so we make an approximate
    // // guess here, and will iteratively fill the buffer until enough samples are added.
    // const int32_t approx_bam_buffer_size = dorado::ssize(resources.devices) * batch_size * std::max(1, (bam_subchunk_len / window_len));

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    const std::vector<polisher::Interval> bam_region_batches =
            create_batches(bam_region_intervals, num_threads,
                           [](const Interval& val) { return val.end - val.start; });

    // std::cerr << "buffer_size = " << buffer_size << "\n";

    // const int64_t notify_buffer_size = dorado::ssize(resources.devices) * batch_size;

    InferenceData buffer;

    // Each iteration of the for loop produces full BAM regions of samples to fit at least num_threads windows.
    // It is important to process full BAM regions because of splitting/merging/splitting and trimming.
    for (const auto [region_id_start, region_id_end] : bam_region_batches) {
        // std::unique_lock<std::mutex> lock_produce(mtx_queue);
        // mtx_queue.wait(lock_produce, [&] {
        //     return (!m_shadow_correction_records.empty() || m_copy_terminate.load());
        // });

        if (region_id_end <= region_id_start) {
            continue;
        }

        const int32_t num_regions = region_id_end - region_id_start;
        const int32_t window_id_start = bam_region_intervals[region_id_start].start;
        const int32_t window_id_end = bam_region_intervals[region_id_end - 1].end;
        const size_t num_windows = static_cast<size_t>(window_id_end - window_id_start);

        // std::cerr << "region_id_start = " << region_id_start << ", region_id_end = " << region_id_end << ", num_regions = " << num_regions << ", num_windows = " << num_windows << "\n";

        // << ", bam_region_intervals.size() = " << bam_region_intervals.size() << "\n";
        // for (int32_t region_id = first; region_id < last; ++region_id) {
        //     std::cerr << "    - region_id = " << region_id << ", bam_region_intervals[region_id].start = " << bam_region_intervals[region_id].start << ", bam_region_intervals[region_id].end = " << bam_region_intervals[region_id].end << "\n";
        // }

        // Encode windows to samples in parallel. Non-const by design, data will be moved.
        std::vector<Sample> region_samples = encode_regions_in_parallel(
                resources.bam_handles, *resources.encoder, draft_lens,
                Span<const Window>(std::data(windows) + window_id_start, num_windows), num_threads);

        spdlog::debug(
                "[producer] Merging the samples into {} BAM chunks. parallel_results.size() = {}",
                num_regions, std::size(region_samples));

        auto [samples, trims] = merge_and_split_bam_regions_in_parallel(
                region_samples, *resources.encoder,
                Span<const Window>(std::data(bam_regions) + region_id_start, num_regions),
                Span<const Interval>(std::data(bam_region_intervals) + region_id_start,
                                     num_regions),
                num_threads, window_len, window_overlap, window_id_start);

        if (std::size(samples) != std::size(trims)) {
            throw std::runtime_error("Size of samples and trims does not match! samples.size() = " +
                                     std::to_string(std::size(samples)) +
                                     ", trims.size() = " + std::to_string(std::size(trims)));
        }

        // Add samples to the batches.
        for (size_t i = 0; i < std::size(samples); ++i) {
            // If any of the samples is of wrong size, create a remainder batch of 1.
            if (dorado::ssize(samples[i].positions_major) != window_len) {
                InferenceData remainder_buffer;
                remainder_buffer.samples = {std::move(samples[i])};
                remainder_buffer.trims = {std::move(trims[i])};
                infer_data.try_push(std::move(remainder_buffer));
                continue;
            }

            // Expand the current buffer.
            buffer.samples.emplace_back(std::move(samples[i]));
            buffer.trims.emplace_back(std::move(trims[i]));

            if (dorado::ssize(buffer.samples) == batch_size) {
                spdlog::debug(
                        "[producer] Pushing a batch of data to infer_data queue. "
                        "buffer.samples.size() = {}",
                        std::size(buffer.samples));
                infer_data.try_push(std::move(buffer));
                buffer = {};
            }
        }

        // buffer.samples.insert(std::end(buffer.samples),
        //                       std::make_move_iterator(std::begin(samples)),
        //                       std::make_move_iterator(std::end(samples)));

        // buffer.trims.insert(std::end(buffer.trims), std::make_move_iterator(std::begin(trims)),
        //                     std::make_move_iterator(std::end(trims)));

        // std::cerr << "samples.size() = " << samples.size() << ", buffer_samples.size() = " << buffer.samples.size() << "\n";

        // // Once the buffer is full enough, add to queue.
        // if (dorado::ssize(buffer.samples) >= notify_buffer_size) {
        //     // This will block until the queue is empty (because it is set to size 1);
        //     // assert(infer_data.capacity() == 1);
        //     spdlog::debug("[producer] Pushing data to infer_data queue. size = {}",
        //                   std::size(buffer.samples));
        //     infer_data.try_push(std::move(buffer));
        //     buffer = {};
        //     spdlog::debug("[producer] Pushed.");
        // }
    }

    if (!std::empty(buffer.samples)) {
        // This will block until the queue is empty (because it is set to size 1);
        // assert(infer_data.capacity() == 1);
        spdlog::debug("[producer] Pushing final batch data to infer_data queue. size = {}",
                      std::size(buffer.samples));
        infer_data.try_push(std::move(buffer));
        buffer = {};
        spdlog::debug("[producer] Pushed final.");
    }

    infer_data.terminate();
}

std::vector<polisher::ConsensusResult> infer_samples_in_parallel_2(
        utils::AsyncQueue<polisher::InferenceData>& batch_queue,
        std::vector<std::shared_ptr<polisher::ModelTorchBase>>& models,
        const polisher::EncoderBase& encoder,
        const polisher::FeatureDecoder& decoder) {
    if (std::empty(models)) {
        throw std::runtime_error("No models have been initialized, cannot run inference.");
    }

    auto batch_infer = [&encoder, &decoder](polisher::ModelTorchBase& model,
                                            const InferenceData& batch) {
        utils::ScopedProfileRange infer("infer", 1);
        timer::TimerHighRes timer_total;

        at::InferenceMode infer_guard;

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        timer::TimerHighRes timer_collate;
        std::vector<torch::Tensor> batch_features;
        batch_features.reserve(std::size(batch.samples));
        for (const auto& sample : batch.samples) {
            batch_features.emplace_back(sample.features);
        }
        torch::Tensor batch_features_tensor = encoder.collate(std::move(batch_features));
        const int64_t time_collate = timer_collate.GetElapsedMilliseconds();

        // Debug output.
        {
            std::ostringstream oss;
            print_tensor_shape(oss, batch_features_tensor);
            spdlog::debug(
                    "About to call forward(): batch_features_tensor.size() = {}, approx "
                    "size: {} MB.",
                    oss.str(),
                    batch_features_tensor.numel() * batch_features_tensor.element_size() /
                            (1024.0 * 1024.0));
        }

        // Inference.
        timer::TimerHighRes timer_forward;
        torch::Tensor output;
        try {
            output = model.predict_on_batch(std::move(batch_features_tensor));
        } catch (std::exception& e) {
            std::cerr << "ERROR! Exception caught: " << e.what() << "\n";
            throw e;
        }
        const int64_t time_forward = timer_forward.GetElapsedMilliseconds();

        // Decode output to bases and qualities.
        // Convert to sequences and qualities.
        timer::TimerHighRes timer_decode;
        std::vector<polisher::ConsensusResult> results = decoder.decode_bases(output);
        const int64_t time_decode = timer_decode.GetElapsedMilliseconds();
        assert(std::size(results) == std::size(batch.samples));

        // Trim the overlapping sequences.
        timer::TimerHighRes timer_trim;
        for (int64_t j = 0; j < dorado::ssize(results); ++j) {
            auto& result = results[j];
            const Sample& sample = batch.samples[j];
            const TrimInfo& trim = batch.trims[j];

            // Trim and mark the region.
            result.draft_id = sample.seq_id;
            result.draft_start = sample.positions_major[trim.start];
            result.draft_end = sample.positions_major[trim.end - 1] + 1;
            result.seq = result.seq.substr(trim.start, trim.end - trim.start);
            result.quals = result.quals.substr(trim.start, trim.end - trim.start);
        }
        const int64_t time_trim = timer_trim.GetElapsedMilliseconds();

        const int64_t time_total = timer_total.GetElapsedMilliseconds();

        spdlog::debug(
                "Computed batch inference. Timings - collate: {} ms, forward: {} ms, decode = {} "
                "ms, trim = {} ms, total = {}",
                time_collate, time_forward, time_decode, time_trim, time_total);

        return results;
    };

    const auto worker = [&](const int32_t tid, std::vector<polisher::ConsensusResult>& results) {
        while (true) {
            spdlog::debug("[consumer {}] Waiting to pop data for inference. Queue size: {}", tid,
                          std::size(batch_queue));

            const auto last_chunk_reserve_time = std::chrono::system_clock::now();

            polisher::InferenceData item;
            const auto pop_status = batch_queue.try_pop_until(
                    item, last_chunk_reserve_time + std::chrono::milliseconds(10000));

            spdlog::debug("[consumer {}] Popped data: item.samples.size() = {}, queue size: {}",
                          tid, std::size(item.samples), batch_queue.size());

            if (pop_status == utils::AsyncQueueStatus::Terminate) {
                break;
            }

            if (pop_status == utils::AsyncQueueStatus::Timeout) {
                spdlog::warn(
                        "[consumer {}] Timeout when popping item from infer_data queue! "
                        "item.samples.size() = {}",
                        tid, std::size(item.samples));
            }

            // This should handle the timeout case too.
            if (std::empty(item.samples)) {
                continue;
            }

            // // Inference.
            std::vector<polisher::ConsensusResult> results_samples =
                    batch_infer(*models[tid], item);

            //         polisher::infer_samples_in_parallel(item.samples, item.trims, resources.models,
            //                                             *resources.encoder, *resources.decoder,
            //                                             opt.window_len, opt.batch_size);

            results.insert(std::end(results), std::make_move_iterator(std::begin(results_samples)),
                           std::make_move_iterator(std::end(results_samples)));
        }
    };

    std::vector<std::vector<polisher::ConsensusResult>> results(std::size(models));

    // const int32_t num_items = static_cast<int32_t>(dorado::ssize(in_samples));
    // One thread per model.
    // const int32_t num_threads = static_cast<int32_t>(dorado::ssize(models));
    // const std::vector<Interval> chunks = compute_chunks(num_items, num_threads);

    // spdlog::info("Starting to call consensus for {} samples using {} devices.", num_items,
    //              std::size(chunks));

    const size_t num_threads = std::size(models);
    cxxpool::thread_pool pool{num_threads};

    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int32_t tid = 0; tid < static_cast<int32_t>(num_threads); ++tid) {
        futures.emplace_back(pool.push(worker, tid, std::ref(results[tid])));
    }

    try {
        for (auto& f : futures) {
            f.get();
        }
    } catch (const std::exception& e) {
        throw std::runtime_error{std::string("Caught exception from inferencetask: ") + e.what()};
    }

    // Flatten the results.
    size_t total_size = 0;
    for (const auto& vals : results) {
        total_size += std::size(vals);
    }
    std::vector<polisher::ConsensusResult> flat_results;
    flat_results.reserve(total_size);
    for (auto& vals : results) {
        flat_results.insert(std::end(flat_results), std::make_move_iterator(std::begin(vals)),
                            std::make_move_iterator(std::end(vals)));
    }

    spdlog::info("Finished running inference.");

    return flat_results;
}

}  // namespace polisher

void run_polishing(const Options& opt, PolisherResources& resources) {
    spdlog::info("Threads: {}", opt.threads);
    spdlog::info("Inference threads: {}", opt.infer_threads);
    spdlog::info("Number of devices: {}", std::size(resources.devices));

    at::InferenceMode infer_guard;

    // Create a .fai index if it doesn't exist.
    const bool rv_fai = utils::create_fai_index(opt.in_draft_fastx_fn);
    if (!rv_fai) {
        spdlog::error("Failed to create/verify a .fai index for input file: '{}'!",
                      opt.in_draft_fastx_fn.string());
        std::exit(EXIT_FAILURE);
    }

    // Load sequence lengths.
    spdlog::info("Loading draft sequence lengths.");
    const std::vector<std::pair<std::string, int64_t>> draft_lens =
            load_seq_lengths(opt.in_draft_fastx_fn);

    // Set the number of threads so that libtorch doesn't cause a thread bomb.
    at::set_num_interop_threads(opt.threads);
    torch::set_num_threads(1);

    // Open the output stream, to std::cout if the path is empty, otherwise to the file.
    auto ofs = get_output_stream(opt.out_consensus_fn);

    // Divide draft sequences into groups of specified size, as sort of a barrier.
    const std::vector<polisher::Interval> draft_batches = polisher::create_batches(
            draft_lens, opt.draft_batch_size,
            [](const std::pair<std::string, int64_t>& val) { return val.second; });

    // Process the draft sequences in batches of user-specified size.
    for (const auto& draft_batch : draft_batches) {
        spdlog::info("=============================");

        // Split the sequences into larger BAM windows, like Medaka.
        spdlog::debug("Creating BAM windows.");
        const std::vector<std::pair<std::string, int64_t>> draft_lens_batch(
                std::begin(draft_lens) + draft_batch.start,
                std::begin(draft_lens) + draft_batch.end);
        const std::vector<polisher::Window> bam_regions = polisher::create_bam_regions(
                draft_lens_batch, opt.bam_chunk, opt.window_overlap, opt.region);

        const int64_t total_bases = std::accumulate(
                std::begin(draft_lens_batch), std::end(draft_lens_batch), static_cast<int64_t>(0),
                [](const int64_t a, const auto& b) { return a + b.second; });
        spdlog::info(
                "Starting to produce consensus for draft sequences: {}-{}/{} (number: {}, total "
                "length: {:.2f} Mbp)",
                draft_batch.start, draft_batch.end, std::size(draft_lens),
                std::size(draft_lens_batch), total_bases / (1000.0 * 1000.0));

        // Each item is one batch for inference.
        utils::AsyncQueue<polisher::InferenceData> batch_queue(opt.num_preloaded_batches);

        std::thread thread_sample_producer = std::thread(
                &polisher::sample_producer, std::ref(resources), std::cref(bam_regions),
                std::cref(draft_lens_batch), opt.threads, opt.batch_size, opt.window_len,
                opt.window_overlap, opt.bam_subchunk, std::ref(batch_queue));

        std::vector<polisher::ConsensusResult> all_results = polisher::infer_samples_in_parallel_2(
                batch_queue, resources.models, *resources.encoder, *resources.decoder);

        // sample_consumer();

        if (thread_sample_producer.joinable()) {
            thread_sample_producer.join();
        }

        // spdlog::debug("[consumer] Consensus results: {}", std::size(all_results));

        // Produce samples (tensors) for inference.
        // auto [samples, trims] = polisher::create_samples(
        //         resources.bam_handles, *resources.encoder, bam_regions, draft_lens_batch,
        //         opt.threads, opt.window_len, opt.window_overlap, opt.bam_subchunk);

        // spdlog::info("Produced num samples: {}", std::size(samples));

        // // Inference.
        // std::vector<polisher::ConsensusResult> results_samples =
        //         polisher::infer_samples_in_parallel(samples, trims, resources.models,
        //                                             *resources.encoder, *resources.decoder,
        //                                             opt.window_len, opt.batch_size);

        spdlog::info(
                "Stitching sequences: {}-{}/{} (number: {}, total "
                "length: {:.2f} Mbp)",
                draft_batch.start, draft_batch.end, std::size(draft_lens),
                std::size(draft_lens_batch), total_bases / (1000.0 * 1000.0));

        // Group samples by sequence ID.
        std::vector<std::vector<std::pair<int64_t, int32_t>>> groups(std::size(draft_lens_batch));
        for (int32_t i = 0; i < dorado::ssize(all_results); ++i) {
            const polisher::ConsensusResult& r = all_results[i];
            groups[r.draft_id].emplace_back(r.draft_start, i);
        }

        // Stitch the windows and write output.
        for (int32_t seq_id = 0; seq_id < static_cast<int32_t>(std::size(groups)); ++seq_id) {
            auto& group = groups[seq_id];
            std::sort(std::begin(group), std::end(group));  // Sort by start pos.

            const polisher::ConsensusResult consensus =
                    polisher::stitch_sequence(opt.in_draft_fastx_fn, draft_lens_batch[seq_id].first,
                                              all_results, group, seq_id);

            const std::string& header = draft_lens_batch[seq_id].first;
            write_consensus_result(*ofs, header, consensus,
                                   (opt.out_format == OutputFormat::FASTQ));
        }
    }

    spdlog::info("Done!");
}

int polish(int argc, char* argv[]) {
    try {
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

        if (opt.verbosity >= 3) {
            spdlog::set_level(spdlog::level::trace);
        } else if (opt.verbosity == 2) {
            spdlog::set_level(spdlog::level::debug);
        } else if (opt.verbosity == 1) {
            spdlog::set_level(spdlog::level::info);
        } else {
            // Pass. No log.
        }

        spdlog::flush_every(std::chrono::seconds(1));

        // Check if input options are good.
        validate_options(opt);

        if (std::empty(opt.model_path)) {
            throw std::runtime_error(
                    "WIP. Currently can only load a model. Not yet fetching a model "
                    "automatically.");
        }

        spdlog::info("Parsing the model config.", opt.threads);
        const std::string model_file = opt.load_scripted_model ? "model.pt" : "weights.pt";
        const polisher::ModelConfig model_config =
                polisher::parse_model_config(opt.model_path / "config.toml", model_file);

        // Create the models, encoders and BAM handles.
        PolisherResources resources =
                create_resources(model_config, opt.in_aln_bam_fn, opt.device_str, opt.threads,
                                 opt.infer_threads, opt.full_precision);

        run_polishing(opt, resources);

    } catch (const std::exception& e) {
        spdlog::error("Caught exception: {}", e.what());
        return EXIT_FAILURE;
    } catch (...) {
        spdlog::error("Caught an unknown exception!");
        return EXIT_FAILURE;
    }

    return 0;
}

}  // namespace dorado
