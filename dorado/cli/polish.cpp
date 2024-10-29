#include "cli/cli_utils.h"
#include "correct/infer.h"
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

// Sample trim_sample(const Sample& sample) {
// }

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

// polisher::ConsensusResult stitch_sequence(
//         const std::filesystem::path& in_draft_fn,
//         const std::string& header,
//         const std::vector<polisher::Sample>& samples,
//         const std::vector<polisher::ConsensusResult>& sample_results,
//         const std::vector<std::pair<int32_t, int32_t>>& samples_for_seq) {

//     // Fetch the draft for gap filling.
//     const std::string draft = fetch_seq(in_draft_fn, header);

//     // Initialize the start/end trim coordinates for each sample.
//     std::vector<Interval> trim_coords;
//     trim_coords.reserve(std::size(samples_for_seq));
//     for (const auto& [_, sample_id]: samples_for_seq) {
//         const polisher::Sample& sample = samples[sample_id];
//         trim_coords.emplace_back(Interval{0, static_cast<int32_t>(sample.positions.size(0))});
//     }

//     polisher::ConsensusResult ret;

//     for (int64_t i = 1; i < static_cast<int64_t>(std::size(sample_results)); ++i) {
//         const auto& sample_id_1 = samples_for_seq[i - 1].second;
//         const auto& sample_id_2 = samples_for_seq[i].second;
//         const polisher::Sample& s1 = samples[sample_id_1];
//         const polisher::Sample& s2 = samples[sample_id_2];

//         const int64_t s1_start = s1.positions.index({0, polisher::MAJOR_COLUMN}).item<int64_t>();
//         const int64_t s1_end = s1.positions.index({-1, polisher::MAJOR_COLUMN}).item<int64_t>() + 1;
//         const int64_t s2_start = s2.positions.index({0, polisher::MAJOR_COLUMN}).item<int64_t>();
//         const int64_t s2_end = s2.positions.index({-1, polisher::MAJOR_COLUMN}).item<int64_t>() + 1;

//         std::cerr << "[stitching i = " << i << "] s1_start = " << s1_start << ", s1_end = " << s1_end << ", s2_start = " << s2_start << ", s2_end = " << s2_end << "\n";
//     }

//     // This won't work well for stitching because it's missing the gap-filled regions, and the trim coordinates need to be
//     // the indices in the positions vector, and not the positions themselves.
//     // Splice the trimmed sequences.
//     for (size_t i = 0; i < std::size(trim_coords); ++i) {
//         const auto& [_, sample_id] = samples_for_seq[i];
//         const polisher::ConsensusResult sample_result = sample_results[sample_id];

//         // Fill the gap if needed.
//         if (i > 0) {
//             const auto& sample_id_1 = samples_for_seq[i - 1].second;
//             const auto& sample_id_2 = samples_for_seq[i].second;
//             const polisher::Sample& s1 = samples[sample_id_1];
//             const polisher::Sample& s2 = samples[sample_id_2];
//             const Interval& trim_1 = trim_coords[i - 1];
//             const Interval& trim_2 = trim_coords[i];
//             const int64_t s1_start = s1.positions.index({trim_1.start, polisher::MAJOR_COLUMN}).item<int64_t>();
//             const int64_t s1_end = s1.positions.index({trim_1.end - 1, polisher::MAJOR_COLUMN}).item<int64_t>() + 1;
//             const int64_t s2_start = s2.positions.index({trim_2.start, polisher::MAJOR_COLUMN}).item<int64_t>();
//             const int64_t s2_end = s2.positions.index({trim_2.end - 1, polisher::MAJOR_COLUMN}).item<int64_t>() + 1;

//             if (s2_start > s1_end) {
//                 const int64_t fill_len = s2_start - s1_end - 1;
//                 ret.seq += draft.substr(s1_end + 1, fill_len);
//                 ret.quals += std::string(fill_len, '!');
//             }
//         }

//         const Interval& trim = trim_coords[i];
//         ret.seq += sample_result.seq.substr(trim.start, trim_coords[i].end - trim_coords[i].start);
//         if (!std::empty(sample_result.quals)) {
//             ret.quals += sample_result.quals.substr(trim_coords[i].start, trim_coords[i].end - trim_coords[i].start);
//         } else {
//             ret.quals += std::string(trim_coords[i].end - trim_coords[i].start, '!');
//         }
//     }

//     return ret;
// }

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
        last_end = sample.positions.index({-1, 0}).item<int64_t>();
        last_i = 0;
    }

    // Iterate over samples in `samples_for_seq`.
    for (size_t i = 1; i < samples_for_seq.size(); ++i) {
        const int sample_index = samples_for_seq[i].second;
        const polisher::Sample& sample = samples[sample_index];
        const polisher::ConsensusResult& result = sample_results[sample_index];

        // Define the start and end positions of the current sample in draft coordinates.
        const int64_t start = sample.positions.index({0, 0}).item<int64_t>();
        const int64_t end = sample.positions.index({-1, 0}).item<int64_t>();

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
            auto prev_sample_positions =
                    samples[samples_for_seq[last_i].second].positions.select(1, 0);
            auto curr_sample_positions = sample.positions.select(1, 0);

            const int64_t prev_sample_mid_idx =
                    torch::searchsorted(prev_sample_positions, overlap_middle, /*right=*/false)
                            .item<int64_t>();
            const int64_t curr_sample_mid_idx =
                    torch::searchsorted(curr_sample_positions, overlap_middle, /*right=*/true)
                            .item<int64_t>();

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

void run_experimental(const Options& opt) {
    std::vector<std::string> devices;
    // int32_t infer_threads = 1;

    // Check the output extension to determine if we need
    // to compute the QVs too.
    const std::string ext = get_lowercase_extension(opt.out_consensus_fn);
    const bool with_quals = ((ext == ".fastq") && (ext != ".fq")) ? true : false;

    spdlog::info("Setting up the device.");

    if (opt.device == "cpu") {
        // infer_threads = 1;
        devices.push_back(opt.device);
    }
#if DORADO_CUDA_BUILD
    else if (utils::starts_with(opt.device, "cuda")) {
        spdlog::info("Parsing CUDA device string.");
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

    spdlog::info("device_str = {}", device_str);

#if DORADO_CUDA_BUILD
    c10::optional<c10::Stream> stream;
    if (device.is_cuda()) {
        spdlog::info("Acquiring a CUDA stream.");
        stream = c10::cuda::getStreamFromPool(false, device.index());
    }
    c10::cuda::OptionalCUDAStreamGuard guard(stream);
#endif

    spdlog::info("Loading the model.");

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

    auto batch_infer = [&gpu_mutexes, &device, &module](
                               const std::vector<polisher::Sample>& samples,
                               const int64_t sample_start, const int64_t sample_end,
                               const int32_t mtx_idx) {
        utils::ScopedProfileRange infer("infer", 1);

        // We can simply stack these since all windows are of the same size. (Smaller windows are set aside.)
        std::vector<torch::Tensor> batch_features;
        for (int64_t i = sample_start; i < sample_end; ++i) {
            batch_features.emplace_back(samples[i].features);
        }
        // torch::Tensor batch_features_tensor = torch::stack(batch_features);
        torch::Tensor batch_features_tensor =
                correction::collate<float>(batch_features, 0.0f, polisher::FeatureTensorType);

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

    const auto process_samples = [&batch_infer](const polisher::CountsFeatureEncoder& encoder,
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
            torch::Tensor output = batch_infer(in_samples, start, end, 0);

            // Convert to sequences and qualities.
            std::vector<polisher::ConsensusResult> new_results =
                    encoder.decode_bases(output, gen_qual);

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
        bam_fset* bam_set = create_bam_fset(opt.in_aln_bam_fn.c_str());

        spdlog::info("Loading draft sequence lengths.");

        const std::vector<std::pair<std::string, int64_t>> draft_lens =
                load_seq_lengths(opt.in_draft_fastx_fn);

        spdlog::info("Creating windows.");

        const auto& [windows, _] = create_windows(draft_lens, opt.window_len, opt.window_overlap);

        spdlog::info("Created {} windows from {} sequences.", std::size(windows),
                     std::size(draft_lens));

        polisher::CountsFeatureEncoder encoder(bam_set);

        spdlog::info("Starting to encode regions for {} windows.", std::size(windows));

        // Encode samples (features). A window can have multiple samples if there was a gap.
        std::vector<polisher::Sample> samples;
        for (int32_t win_id = 0; win_id < static_cast<int32_t>(std::size(windows)); ++win_id) {
            const auto& window = windows[win_id];
            const std::string& name = draft_lens[window.seq_id].first;
            std::vector<polisher::Sample> new_samples =
                    encoder.encode_region(name, window.start, window.end, window.seq_id, win_id);

            samples.insert(std::end(samples), std::make_move_iterator(std::begin(new_samples)),
                           std::make_move_iterator(std::end(new_samples)));
        }

        spdlog::info("Processing samples in batches. Num samples: {}.", std::size(samples));

        const std::vector<polisher::ConsensusResult> results_samples =
                process_samples(encoder, samples, opt.batch_size, with_quals);

        if (std::size(results_samples) != std::size(samples)) {
            throw std::runtime_error{
                    "Wrong number of results for input samples! std::size(results_samples) = " +
                    std::to_string(std::size(results_samples)) +
                    ", std::size(samples) = " + std::to_string(std::size(samples))};
        }

        // for (size_t i = 0; i < std::size(results_samples); ++i) {
        //     std::cerr << "[results_samples i = " << i << "] seq_id = " << samples[i].seq_id
        //               << ", win_id = " << samples[i].window_id
        //               << ", seq len = " << std::size(results_samples[i].seq)
        //               << ", seq: " << results_samples[i].seq << "\n";
        // }

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

            // for (size_t i = 0; i < std::size(data); ++i) {
            //     std::cerr << "[seq_id = " << seq_id << ", i = " << i
            //               << "] start = " << data[i].first << ", sample_id = " << data[i].second
            //               << "\n";
            // }

            const polisher::ConsensusResult consensus =
                    stitch_sequence(opt.in_draft_fastx_fn, draft_lens[seq_id].first, samples,
                                    results_samples, data);

            const std::string header = std::string("consensus") + "-" + draft_lens[seq_id].first;
            write_seq(ofs, header, consensus, with_quals);
        }

        destroy_bam_fset(bam_set);
    }

    // for (size_t i = 0; i < std::size(results_remainders); ++i) {
    //     std::cerr << "[results_remainders i = " << i << "] len = " << std::size(results_remainders[i].seq) << ", seq: " << results_remainders[i].seq << "\n";
    // }

    // // Inference.
    // // Process each sample individually in batch size 1 for now, to avoid
    // // differing sample lengths.
    // // This should be done in batches later.
    // std::vector<std::vector<torch::Tensor>> outputs;
    // outputs.resize(std::size(samples));
    // for (size_t win_id = 0; win_id < std::size(samples); ++win_id) {
    //     outputs[win_id].reserve(std::size(samples[win_id]));
    //     for (size_t j = 0; j < std::size(samples[win_id]); ++j) {
    //         std::vector<polisher::Sample> samples_to_infer{samples[win_id][j]};
    //         torch::Tensor output = batch_infer(std::move(samples_to_infer), 0);
    //         outputs[win_id].emplace_back(std::move(output));
    //     }
    // }

    // std::vector<std::vector<polisher::ConsensusResult>> results;
    // results.resize(std::size(samples));
    // for (size_t win_id = 0; win_id < std::size(samples); ++win_id) {
    //     results[win_id].reserve(std::size(samples[win_id]));
    //     for (size_t j = 0; j < std::size(outputs[win_id]); ++j) {
    //         std::vector<polisher::ConsensusResult> result =
    //                 encoder.decode_bases(outputs[win_id][j], with_quals);
    //         results[win_id].insert(std::end(results[win_id]),
    //                                std::make_move_iterator(std::begin(result)),
    //                                std::make_move_iterator(std::end(result)));
    //     }
    // }

    // for (const Interval interval: seq_window_ranges) {
    //     for (int32_t i = interval.start; i < interval.end; ++i) {
    //         const Window& window = windows[i];
    //     }
    // }

    // for (const auto& window : windows) {
    //     polisher::CountsFeatureEncoder encoder(bam_set);
    //     std::vector<polisher::Sample> samples =
    //             encoder.encode_region(window.name, window.start, window.end);

    //     std::vector<polisher::Sample> samples_to_infer;
    //     std::vector<polisher::Sample> remainders;
    //     for (auto& sample : samples) {
    //         // if (static_cast<int32_t>(std::size(sample.positions)) < opt.window_len) {
    //         //     remainders.emplace_back(std::move(sample));
    //         //     continue;
    //         // }
    //         samples_to_infer.emplace_back(std::move(sample));
    //     }

    //     spdlog::info("Inference on batch size: {}", std::size(samples_to_infer));
    //     spdlog::info("Remainders size: {}", remainders.size());

    //     const torch::Tensor output = batch_infer(samples_to_infer, 0);

    //     std::vector<polisher::ConsensusResult> results =
    //             encoder.decode_bases(output, with_quals);

    //     // Write output.
    //     {
    //         for (size_t i = 0; i < std::size(results); ++i) {
    //             std::string& seq = results[i].seq;
    //             std::string& quals = results[i].quals;

    //             size_t n = 0;
    //             for (size_t j = 0; j < std::size(seq); ++j) {
    //                 if (seq[j] == '*') {
    //                     continue;
    //                 }
    //                 seq[n] = seq[j];
    //                 quals[n] = quals[j];
    //                 ++n;
    //             }
    //             seq.resize(n);
    //             quals.resize(n);
    //             // seq.erase(std::remove(seq.begin(), seq.end(), '*'), seq.end());

    //             if (with_quals) {
    //                 ofs << "@consensus-" << window.name << ':' << (window.start + 1) << '-'
    //                     << window.end << '\n'
    //                     << results[i].seq << "\n+\n"
    //                     << results[i].quals << '\n';
    //             } else {
    //                 ofs << ">consensus-" << window.name << ':' << (window.start + 1) << '-'
    //                     << window.end << '\n'
    //                     << seq << '\n';
    //             }
    //         }
    //     }
    // }

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
