#include "cli/cli_utils.h"
#include "correct/infer.h"
#include "dorado_version.h"
#include "model_downloader/model_downloader.h"
#include "polish/features.h"
#include "polish/medaka_bamiter.h"
#include "polish/medaka_counts.h"
#include "polish/model.h"
#include "polish/polish_models.h"
#include "polish/sample.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/gpu_profiling.h"
#include "utils/arg_parse_ext.h"
#include "utils/fai_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"

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
#include <unordered_set>
#include <vector>
using namespace std::chrono_literals;

#ifndef _WIN32
#include <unistd.h>
#endif

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

std::vector<DeviceInfo> init_devices(const std::string& devices_str) {
    std::vector<DeviceInfo> devices;

    if (devices_str == "cpu") {
        // infer_threads = 1;
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

const std::vector<std::pair<int32_t, int32_t>> compute_chunks(const int32_t num_items,
                                                              const int32_t num_chunks) {
    std::vector<std::pair<int32_t, int32_t>> chunks;
    const int32_t chunk_size = num_items / num_chunks;
    std::vector<int32_t> chunk_sizes(num_chunks, chunk_size);
    for (int32_t i = 0; i < (num_items % num_chunks); ++i) {
        ++chunk_sizes[i];
    }
    int32_t sum = 0;
    for (const int32_t v : chunk_sizes) {
        if (v == 0) {
            continue;
        }
        chunks.emplace_back(sum, sum + v);
        sum += v;
    }
    if (sum != num_items) {
        throw std::runtime_error{
                "Wrong sum of items divided into chunks! num_items = " + std::to_string(num_items) +
                ", num_chunks = " + std::to_string(num_chunks) + ", sum = " + std::to_string(sum)};
    }
    return chunks;
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

struct Window {
    // std::string name;
    int32_t seq_id = -1;
    int64_t start = 0;
    int64_t end = 0;
    int64_t length = 0;
    int32_t window_id = 0;
    int32_t num_windows = 0;
};

struct Interval {
    int32_t start = 0;
    int32_t end = 0;
};

/**
 * \brief Linearly splits sequence lengths into windows. It also returns the backward mapping of which
 *          windows correspond to which sequences, needed for stitching.
 */
std::pair<std::vector<Window>, std::vector<Interval>> create_windows(
        const std::vector<std::pair<std::string, int64_t>>& seq_lens,
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
    std::vector<Interval> window_ranges;
    for (int32_t seq_id = 0; seq_id < static_cast<int32_t>(std::size(seq_lens)); ++seq_id) {
        const auto& [name, length] = seq_lens[seq_id];

        const int32_t num_windows =
                static_cast<int32_t>(std::ceil(static_cast<double>(length) / window_len));

        ret.reserve(std::size(ret) + num_windows);
        const int32_t interval_start = static_cast<int32_t>(std::size(ret));

        int32_t win_id = 0;
        for (int64_t start = 0; start < length; start += (window_len - window_overlap), ++win_id) {
            const int64_t end = std::min(length, start + window_len);
            ret.emplace_back(Window{seq_id, start, end, length, win_id, num_windows});
            if (end == length) {
                break;
            }
        }

        window_ranges.emplace_back(Interval{interval_start, static_cast<int32_t>(std::size(ret))});
    }

    return {ret, window_ranges};
}

std::string fetch_seq(const std::filesystem::path& index_fn,
                      const std::string& seq_name,
                      int32_t start = 0,
                      int32_t end = -1) {
    faidx_t* fai = fai_load(index_fn.c_str());
    if (!fai) {
        spdlog::error("Failed to load index for file: '{}'.", index_fn.string());
        return {};
    }

    const int32_t seq_len = faidx_seq_len(fai, seq_name.c_str());

    start = std::max(start, 0);
    end = (end < 0) ? seq_len : std::min(end, seq_len);

    int32_t temp_seq_len = 0;
    char* seq = faidx_fetch_seq(fai, seq_name.c_str(), start, end - 1, &temp_seq_len);

    if (end <= start) {
        spdlog::error(
                "Cannot load sequence because end <= start! seq_name = {}, start = {}, end = {}.",
                seq_name, start, end);
        return {};
    }

    if (temp_seq_len != (end - start)) {
        spdlog::error(
                "Loaded sequence length does not match the specified interval! seq_name = {}, "
                "start = {}, end = {}, loaded len = {}.",
                seq_name, start, end, temp_seq_len);
        return {};
    }

    std::string ret;
    if (seq) {
        ret = std::string(seq, temp_seq_len);
        free(seq);
    }

    fai_destroy(fai);

    return ret;
}

polisher::ConsensusResult stitch_sequence(
        const std::filesystem::path& in_draft_fn,
        const std::string& header,
        const std::vector<polisher::Sample>& samples,
        const std::vector<polisher::ConsensusResult>& sample_results,
        const std::vector<std::pair<int32_t, int32_t>>& samples_for_seq) {
    if (std::empty(samples_for_seq)) {
        return {};
    }

    // Fetch the draft sequence for gap filling.
    const std::string draft = fetch_seq(in_draft_fn, header);

    // Initialize output consensus sequence and quality strings.
    std::string consensus_seq;
    std::string consensus_quals;

    // Track the end position of the last processed sample.
    int64_t last_end = -1;
    size_t last_i = -1;

    // Append the entire sequence of the current sample if no overlap was found.
    {
        const int sample_index = samples_for_seq[0].second;
        const polisher::Sample& sample = samples[sample_index];
        const polisher::ConsensusResult& result = sample_results[sample_index];
        consensus_seq = result.seq;
        consensus_quals = result.quals;
        last_end = sample.positions_major.back();
        last_i = 0;
    }

    // Iterate over samples in `samples_for_seq`.
    for (size_t i = 1; i < samples_for_seq.size(); ++i) {
        const int sample_index = samples_for_seq[i].second;
        const polisher::Sample& sample = samples[sample_index];
        const polisher::ConsensusResult& result = sample_results[sample_index];

        // Define the start and end positions of the current sample in draft coordinates.
        const int64_t start = sample.positions_major.front();
        const int64_t end = sample.positions_major.back();

        // std::cerr << "[stitching i = " << i << "] samples_for_seq.size() = " << samples_for_seq.size()
        //     << ", consensus_seq.size() = "
        //     << consensus_seq.size()
        //     << ", consensus_quals.size() = " << consensus_quals.size()
        //     << ", last_end = " << last_end
        //     << ", start = " << start << ", end = " << end << "\n";

        if (end < last_end) {
            continue;
        }

        if (last_end >= start) {
            // Compute midpoint of overlap.
            const int64_t overlap_middle = (last_end + start) / 2;

            // Find midpoint indices using searchsorted.
            const std::vector<int64_t>& prev_sample_positions =
                    samples[samples_for_seq[last_i].second]
                            .positions_major;  // .select(1, 0).contiguous();
            const std::vector<int64_t>& curr_sample_positions =
                    sample.positions_major;  // .select(1, 0).contiguous();

            // const int64_t prev_sample_mid_idx =
            //         torch::searchsorted(prev_sample_positions, overlap_middle, /*right=*/false)
            //                 .item<int64_t>();
            // const int64_t curr_sample_mid_idx =
            //         torch::searchsorted(curr_sample_positions, overlap_middle, /*right=*/true)
            //                 .item<int64_t>();

            // Find the index for prev_sample_positions (equivalent to right=false)
            const auto prev_sample_mid_iter = std::lower_bound(
                    prev_sample_positions.begin(), prev_sample_positions.end(), overlap_middle);
            const int64_t prev_sample_mid_idx =
                    std::distance(prev_sample_positions.begin(), prev_sample_mid_iter);

            // Find the index for curr_sample_positions (equivalent to right=true)
            const auto curr_sample_mid_iter = std::upper_bound(
                    curr_sample_positions.begin(), curr_sample_positions.end(), overlap_middle);
            const int64_t curr_sample_mid_idx =
                    std::distance(curr_sample_positions.begin(), curr_sample_mid_iter);

            // Trim the previous consensus to avoid the overlap.
            const int64_t num_to_remove =
                    static_cast<int64_t>(
                            std::size(sample_results[samples_for_seq[last_i].second].seq)) -
                    prev_sample_mid_idx;
            // std::cerr << "prev_sample_mid_idx = " << prev_sample_mid_idx << ", curr_sample_mid_idx = " << curr_sample_mid_idx
            //     << ", num_to_remove = " << num_to_remove << "\n";
            consensus_seq.resize(consensus_seq.size() - num_to_remove);
            consensus_quals.resize(consensus_quals.size() - num_to_remove);
            // consensus_seq.erase(consensus_seq.size() - (last_end - overlap_middle + 1));
            // consensus_quals.erase(consensus_quals.size() - (last_end - overlap_middle + 1));

            // Append non-overlapping portion of current sample.
            consensus_seq += result.seq.substr(curr_sample_mid_idx);
            consensus_quals += result.quals.substr(curr_sample_mid_idx);

        } else if (start > (last_end + 1)) {
            // Gap case between the previous sample and the current sample.
            const int64_t gap_start = last_end + 1;
            const int64_t gap_end = start;

            // Fill gap with draft sequence and low-quality placeholders.
            consensus_seq += draft.substr(gap_start, gap_end - gap_start);
            consensus_quals += std::string(gap_end - gap_start, '!');
        }

        // // Append the entire sequence of the current sample if no overlap was found.
        // consensus_seq += result.seq;
        // consensus_quals += result.quals;

        // Update the last processed end position.
        last_end = end;
        last_i = i;
    }

    return polisher::ConsensusResult{consensus_seq, consensus_quals};
}

std::vector<polisher::Sample> create_samples(
        const std::filesystem::path& in_aln_bam_fn,
        const std::vector<std::pair<std::string, int64_t>>& draft_lens,
        const std::vector<Window>& windows,
        const int32_t num_threads,
        const int32_t window_len,
        const int32_t window_overlap) {
    // Open the BAM file for each thread and spawn encoders.
    spdlog::info("Creating {} encoders.", num_threads);
    std::vector<bam_fset*> bam_sets;
    std::vector<polisher::CountsFeatureEncoder> encoders;
    for (int32_t i = 0; i < num_threads; ++i) {
        bam_sets.emplace_back(create_bam_fset(in_aln_bam_fn.c_str()));
        encoders.emplace_back(polisher::CountsFeatureEncoder(bam_sets.back()));
    }

    const auto worker_samples = [&](const int32_t thread_id, const int32_t start, const int32_t end,
                                    std::vector<std::vector<polisher::Sample>>& results) {
        for (int32_t i = start; i < end; ++i) {
            const auto& window = windows[i];
            const std::string& name = draft_lens[window.seq_id].first;
            if (thread_id == 0) {
                spdlog::info(
                        "Processing i = {}, start = {}, end = {}, region = "
                        "{}:{}-{} ({} %).",
                        i, start, end, name, window.start, window.end,
                        100.0 * static_cast<double>(i - start) / (end - start));
            }
            results[i] = encoders[thread_id].encode_region(
                    name, window.start, window.end, window.seq_id, i, window_len, window_overlap);
        }
    };

    std::vector<std::vector<polisher::Sample>> win_results(std::size(windows));

    const int32_t num_items = static_cast<int32_t>(std::size(windows));
    const std::vector<std::pair<int32_t, int32_t>> chunks = compute_chunks(num_items, num_threads);

    for (size_t i = 0; i < std::size(chunks); ++i) {
        const auto& [start, end] = chunks[i];
        std::cerr << "[chunk i = " << i << "] start = " << start << ", end = " << end << "\n";
    }

    spdlog::info("Starting to encode regions for {} windows using {} threads.", std::size(windows),
                 std::size(chunks));

    cxxpool::thread_pool pool{std::size(chunks)};

    std::vector<std::future<void>> futures;
    futures.reserve(std::size(chunks));
    for (int32_t tid = 0; tid < static_cast<int32_t>(std::size(chunks)); ++tid) {
        const auto [chunk_start, chunk_end] = chunks[tid];
        futures.emplace_back(
                pool.push(worker_samples, tid, chunk_start, chunk_end, std::ref(win_results)));
    }
    for (auto& f : futures) {
        f.wait();
    }

    spdlog::info("Flattening the samples.");

    // Flatten the samples.
    size_t num_samples = 0;
    for (const auto& win_samples : win_results) {
        num_samples += std::size(win_samples);
    }

    std::vector<polisher::Sample> samples;
    samples.reserve(num_samples);
    for (auto& win_samples : win_results) {
        samples.insert(std::end(samples), std::make_move_iterator(std::begin(win_samples)),
                       std::make_move_iterator(std::end(win_samples)));
    }

    for (auto& bam_set : bam_sets) {
        destroy_bam_fset(bam_set);
    }

    return samples;
}

}  // namespace

void run_polishing(const Options& opt, const std::vector<DeviceInfo>& devices) {
    if (std::empty(devices)) {
        spdlog::error("Zero devices initialized! Need at least one device to run.");
        std::exit(EXIT_FAILURE);
    }

    // Check the output extension to determine if we need
    // to compute the QVs too.
    const std::string ext = get_lowercase_extension(opt.out_consensus_fn);
    const bool with_quals = ((ext == ".fastq") && (ext != ".fq")) ? true : false;

    const DeviceInfo& device_info = devices.front();

    spdlog::info("Using: device_str = {}", device_info.name);

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (device.is_cuda()) {
        spdlog::info("Acquiring a CUDA stream.");
        stream = c10::cuda::getStreamFromPool(false, device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    at::InferenceMode infer_guard;

    std::array<std::mutex, 32> gpu_mutexes;  // One per GPU.

    auto batch_infer = [](polisher::TorchModel& model, const std::vector<polisher::Sample>& samples,
                          const int64_t sample_start, const int64_t sample_end,
                          std::mutex& gpu_mutex) {
        utils::ScopedProfileRange infer("infer", 1);

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        std::vector<torch::Tensor> batch_features;
        for (int64_t i = sample_start; i < sample_end; ++i) {
            std::cout << "[i = " << i
                      << "] sample.positions = " << samples[i].positions_major.front() << " - "
                      << samples[i].positions_major.back() << "\n";
            batch_features.emplace_back(samples[i].features);
        }
        // torch::Tensor batch_features_tensor = torch::stack(batch_features);
        torch::Tensor batch_features_tensor =
                correction::collate<float>(batch_features, 0.0f, polisher::FeatureTensorType);

        spdlog::info(
                "About to call forward(): batch_features_tensor.size() = ({}, {}, {}), approx "
                "size: {} MB.",
                batch_features_tensor.size(0), batch_features_tensor.size(1),
                batch_features_tensor.size(2),
                batch_features_tensor.numel() * batch_features_tensor.element_size() /
                        (1024.0 * 1024.0));

        std::unique_lock<std::mutex> lock(gpu_mutex);

        torch::Tensor output;
        try {
            output = model.predict_on_batch(std::move(batch_features_tensor));
        } catch (std::exception& e) {
            throw e;
        }
        lock.unlock();

        return output;
    };

    const auto process_samples = [&batch_infer, &gpu_mutexes](
                                         polisher::TorchModel& model,
                                         const polisher::CountsFeatureDecoder& decoder,
                                         const std::vector<polisher::Sample>& in_samples,
                                         const int32_t batch_size, const bool gen_qual) {
        /**
         * \brief This creates a copy of the features from samples, so we have the original ones for trimming.
         */
        // std::vector<torch::Tensor> outputs;
        std::vector<polisher::ConsensusResult> results;
        const int64_t num_samples = static_cast<int64_t>(std::size(in_samples));
        for (int64_t start = 0; start < num_samples; start += batch_size) {
            const int64_t end = std::min((start + batch_size), num_samples);

            // Inference.
            torch::Tensor output = batch_infer(model, in_samples, start, end, gpu_mutexes[0]);

            // Convert to sequences and qualities.
            std::vector<polisher::ConsensusResult> new_results =
                    decoder.decode_bases(output, gen_qual);

            assert(static_cast<int64_t>(std::size(new_results)) == (end - start));

            // Trim the padding from the back of each sequence, and append.
            for (int64_t j = 0; j < static_cast<int64_t>(std::size(new_results)); ++j) {
                auto& result = new_results[j];
                const int64_t actual_size = in_samples[start + j].features.size(0);
                result.seq.resize(actual_size);
                result.quals.resize(actual_size);
                results.emplace_back(std::move(result));
            }

            spdlog::info(
                    "Processed another batch of {} samples. Total samples processed: {}, "
                    "num_samples = {}.",
                    (end - start), end, num_samples);
        }

        assert(std::size(results) == std::size(in_samples));

        return results;
    };

    const auto write_seq = [](std::ostream& os, const std::string& seq_name,
                              const polisher::ConsensusResult& result, const bool write_quals) {
        if (std::empty(result.seq)) {
            return;
        }

        std::string seq(std::size(result.seq), '\0');
        std::string quals(std::size(result.seq), '!');

        size_t n = 0;
        for (size_t j = 0; j < std::size(result.seq); ++j) {
            if (result.seq[j] == '*') {
                continue;
            }
            seq[n] = result.seq[j];
            if (!std::empty(result.quals)) {
                quals[n] = result.quals[j];
            }
            ++n;
        }
        seq.resize(n);
        quals.resize(n);

        if (write_quals) {
            os << '@' << seq_name << '\n' << seq << "\n+\n" << quals << '\n';
        } else {
            os << '>' << seq_name << '\n' << seq << '\n';
        }
    };

    // Main processing code.
    {
        spdlog::info("Loading draft sequence lengths.");

        const std::vector<std::pair<std::string, int64_t>> draft_lens =
                load_seq_lengths(opt.in_draft_fastx_fn);

        spdlog::info("Creating windows.");

        // Create BAM windows (regions) to create pileup. The features (samples) will
        // be split further into windows of window_len in size prior to inference.
        const std::vector<Window> windows =
                create_windows(draft_lens, opt.bam_chunk, opt.window_overlap).first;

        spdlog::info("Created {} windows from {} sequences.", std::size(windows),
                     std::size(draft_lens));

        // IMPORTANT: The intra-thread parallelism was killing performance when multiple threads were used.
        //              Remember to reset this at the inference stage so that the CPU-only runs don't suffer.
        torch::set_num_threads(1);
        at::set_num_interop_threads(opt.threads);

        // Encode samples (features) in parallel. A window can have multiple samples if there was a gap.
        std::vector<polisher::Sample> samples =
                create_samples(opt.in_aln_bam_fn, draft_lens, windows, opt.threads, opt.window_len,
                               opt.window_overlap);

        // Increase the number of threads again for inter-op parallelism.
        torch::set_num_threads(opt.threads);

        // Construct the model.
        spdlog::info("Loading the model.");
        const std::unique_ptr<polisher::TorchModel> model =
                create_model(opt.model_path / "model.pt", device_info);

        spdlog::info("Processing samples in batches. Num samples: {}.", std::size(samples));

        const polisher::CountsFeatureDecoder decoder;

        const std::vector<polisher::ConsensusResult> results_samples =
                process_samples(*model, decoder, samples, opt.batch_size, with_quals);

        if (std::size(results_samples) != std::size(samples)) {
            throw std::runtime_error{
                    "Wrong number of results for input samples! std::size(results_samples) = " +
                    std::to_string(std::size(results_samples)) +
                    ", std::size(samples) = " + std::to_string(std::size(samples))};
        }

        // Stitching information, collect all samples for each sequence.
        std::vector<std::vector<std::pair<int32_t, int32_t>>> samples_for_seqs(
                std::size(draft_lens));
        for (int32_t i = 0; i < static_cast<int32_t>(std::size(samples)); ++i) {
            const polisher::Sample& sample = samples[i];
            samples_for_seqs[sample.seq_id].emplace_back(sample.region_start, i);
        }

        std::ofstream ofs(opt.out_consensus_fn);

        // Stitch the windows.
        for (size_t seq_id = 0; seq_id < std::size(samples_for_seqs); ++seq_id) {
            auto& data = samples_for_seqs[seq_id];

            if (std::empty(data)) {
                continue;
            }

            // Sort by region start position, for every sequence.
            std::sort(std::begin(data), std::end(data));

            const polisher::ConsensusResult consensus =
                    stitch_sequence(opt.in_draft_fastx_fn, draft_lens[seq_id].first, samples,
                                    results_samples, data);

            const std::string header = std::string("consensus") + "-" + draft_lens[seq_id].first;
            write_seq(ofs, header, consensus, with_quals);
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

    // Initialize the log level.
    if (opt.verbosity) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(opt.verbosity));
    }

    spdlog::set_level(spdlog::level::info);

    spdlog::flush_every(std::chrono::seconds(1));  // flush every 3 seconds

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
