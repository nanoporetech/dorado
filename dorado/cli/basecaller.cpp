#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#include "nn/CRFModel.h"
#include "utils/basecaller_utils.h"
#include "utils/models.h"
#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD
#include "nn/ModBaseRunner.h"
#include "nn/ModelRunner.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/StatsCounter.h"
#include "utils/bam_utils.h"
#include "utils/cli_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace dorado {

using HtsWriter = utils::HtsWriter;
using HtsReader = utils::HtsReader;
using dorado::utils::default_parameters;

void setup(std::vector<std::string> args,
           const std::filesystem::path& model_path,
           const std::string& data_path,
           const std::string& remora_models,
           const std::string& device,
           const std::string& ref,
           size_t chunk_size,
           size_t overlap,
           size_t batch_size,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           float methylation_threshold_pct,
           HtsWriter::OutputMode output_mode,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           std::string read_list_file_path,
           bool recursive_file_loading,
           int kmer_size,
           int window_size,
           uint64_t mm2_index_batch_size,
           bool skip_model_compatibility_check) {
    torch::set_num_threads(1);
    std::vector<Runner> runners;

    // Default is 1 device.  CUDA path may alter this.
    int num_devices = 1;

    if (device == "cpu") {
        num_runners = std::thread::hardware_concurrency();
        if (batch_size == 0) {
            batch_size = 128;
        }
        spdlog::debug("- CPU calling: set batch size to {}, num_runners to {}", batch_size,
                      num_runners);

        for (size_t i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(model_path, device,
                                                                        chunk_size, batch_size));
        }
    }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
    else if (device == "metal") {
        auto caller = create_metal_caller(model_path, chunk_size, batch_size);
        for (size_t i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<MetalModelRunner>(caller));
        }
        if (runners.back()->batch_size() != batch_size) {
            spdlog::debug("- set batch size to {}", runners.back()->batch_size());
        }
    } else {
        throw std::runtime_error(std::string("Unsupported device: ") + device);
    }
#else   // ifdef __APPLE__
    else {
        auto devices = utils::parse_cuda_device_string(device);
        num_devices = devices.size();
        if (num_devices == 0) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
        for (auto device_string : devices) {
            auto caller = create_cuda_caller(model_path, chunk_size, batch_size, device_string);
            for (size_t i = 0; i < num_runners; i++) {
                runners.push_back(std::make_shared<CudaModelRunner>(caller));
            }
            if (runners.back()->batch_size() != batch_size) {
                spdlog::debug("- set batch size for {} to {}", device_string,
                              runners.back()->batch_size());
            }
        }
    }
#endif  // __APPLE__
#endif  // DORADO_GPU_BUILD

    // verify that all runners are using the same stride, in case we allow multiple models in future
    auto model_stride = runners.front()->model_stride();
    auto adjusted_chunk_size = runners.front()->chunk_size();
    assert(std::all_of(runners.begin(), runners.end(), [&](auto runner) {
        return runner->model_stride() == model_stride &&
               runner->chunk_size() == adjusted_chunk_size;
    }));

    if (chunk_size != adjusted_chunk_size) {
        spdlog::debug("- adjusted chunk size to match model stride: {} -> {}", chunk_size,
                      adjusted_chunk_size);
        chunk_size = adjusted_chunk_size;
    }
    auto adjusted_overlap = (overlap / model_stride) * model_stride;
    if (overlap != adjusted_overlap) {
        spdlog::debug("- adjusted overlap to match model stride: {} -> {}", overlap,
                      adjusted_overlap);
        overlap = adjusted_overlap;
    }

    if (!remora_models.empty() && output_mode == HtsWriter::OutputMode::FASTQ) {
        throw std::runtime_error("Modified base models cannot be used with FASTQ output");
    }

    if (!ref.empty() && output_mode == HtsWriter::OutputMode::FASTQ) {
        throw std::runtime_error("Alignment to reference cannot be used with FASTQ output.");
    }

    std::vector<std::filesystem::path> remora_model_list;
    std::istringstream stream{remora_models};
    std::string model;
    while (std::getline(stream, model, ',')) {
        remora_model_list.push_back(model);
    }

    // generate model callers before nodes or it affects the speed calculations
    std::vector<std::unique_ptr<ModBaseRunner>> remora_runners;
    std::vector<std::string> modbase_devices;
#if DORADO_GPU_BUILD && !defined(__APPLE__)
    if (device != "cpu") {
        modbase_devices = utils::parse_cuda_device_string(device);
    } else
#endif
    {
        modbase_devices.push_back(device);
    }
    for (const auto& device_string : modbase_devices) {
        auto caller = create_modbase_caller(remora_model_list, remora_batch_size, device_string);
        for (size_t i = 0; i < default_parameters.remora_runners_per_caller; i++) {
            remora_runners.push_back(std::make_unique<ModBaseRunner>(caller));
        }
    };

    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    auto read_groups = DataLoader::load_read_groups(data_path, model_name, recursive_file_loading);

    auto read_list = utils::load_read_list(read_list_file_path);

    // Check sample rate of model vs data.
    auto data_sample_rate = DataLoader::get_sample_rate(data_path, recursive_file_loading);
    auto model_sample_rate = get_model_sample_rate(model_path);
    if (!skip_model_compatibility_check && (data_sample_rate != model_sample_rate)) {
        std::stringstream err;
        err << "Sample rate for model (" << model_sample_rate << ") and data (" << data_sample_rate
            << ") don't match.";
        throw std::runtime_error(err.str());
    }

    size_t num_reads = DataLoader::get_num_reads(data_path, read_list, recursive_file_loading);
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    bool rna = utils::is_rna_model(model_path), duplex = false;

    auto const thread_allocations = utils::default_thread_allocations(
            num_devices, !remora_model_list.empty() ? num_remora_threads : 0);

    std::unique_ptr<sam_hdr_t, void (*)(sam_hdr_t*)> hdr(sam_hdr_init(), sam_hdr_destroy);
    utils::add_pg_hdr(hdr.get(), args);
    utils::add_rg_hdr(hdr.get(), read_groups);
    std::shared_ptr<HtsWriter> bam_writer;
    std::shared_ptr<utils::Aligner> aligner;
    MessageSink* converted_reads_sink = nullptr;
    if (ref.empty()) {
        bam_writer = std::make_shared<HtsWriter>("-", output_mode,
                                                 thread_allocations.writer_threads, num_reads);
        bam_writer->write_header(hdr.get());
        converted_reads_sink = bam_writer.get();
    } else {
        bam_writer = std::make_shared<HtsWriter>("-", output_mode,
                                                 thread_allocations.writer_threads, num_reads);
        aligner = std::make_shared<utils::Aligner>(*bam_writer, ref, kmer_size, window_size,
                                                   mm2_index_batch_size,
                                                   thread_allocations.aligner_threads);
        utils::add_sq_hdr(hdr.get(), aligner->get_sequence_records_for_header());
        bam_writer->write_header(hdr.get());
        converted_reads_sink = aligner.get();
    }
    ReadToBamType read_converter(*converted_reads_sink, emit_moves, rna,
                                 thread_allocations.read_converter_threads,
                                 methylation_threshold_pct);
    StatsCounterNode stats_node(read_converter, duplex);
    ReadFilterNode read_filter_node(stats_node, min_qscore, default_parameters.min_seqeuence_length,
                                    thread_allocations.read_filter_threads);

    std::unique_ptr<ModBaseCallerNode> mod_base_caller_node;
    MessageSink* basecaller_node_sink = static_cast<MessageSink*>(&read_filter_node);
    if (!remora_model_list.empty()) {
        mod_base_caller_node = std::make_unique<ModBaseCallerNode>(
                read_filter_node, std::move(remora_runners),
                thread_allocations.remora_threads * num_devices, model_stride, remora_batch_size);
        basecaller_node_sink = static_cast<MessageSink*>(mod_base_caller_node.get());
    }
    const int kBatchTimeoutMS = 100;
    BasecallerNode basecaller_node(*basecaller_node_sink, std::move(runners), overlap,
                                   kBatchTimeoutMS, model_name);
    ScalerNode scaler_node(basecaller_node, thread_allocations.scaler_node_threads);

    DataLoader loader(scaler_node, "cpu", thread_allocations.loader_threads, max_reads, read_list);

    loader.load_reads(data_path, recursive_file_loading);

    bam_writer->join();
    stats_node.dump_stats();
}

int basecaller(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("model").help("the basecaller model to run.");

    parser.add_argument("data").help("the data directory.");

    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\", \"cpu\" "
                  "etc..")
            .default_value(default_parameters.device);

    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to basecall. If not provided, all "
                  "reads will be basecalled")
            .default_value(std::string(""));

    parser.add_argument("-n", "--max-reads").default_value(0).scan<'i', int>();

    parser.add_argument("--min-qscore").default_value(0).scan<'i', int>();

    parser.add_argument("-b", "--batchsize")
            .default_value(default_parameters.batchsize)
            .scan<'i', int>()
            .help("if 0 an optimal batchsize will be selected");

    parser.add_argument("-c", "--chunksize")
            .default_value(default_parameters.chunksize)
            .scan<'i', int>();

    parser.add_argument("-o", "--overlap")
            .default_value(default_parameters.overlap)
            .scan<'i', int>();

    parser.add_argument("-r", "--recursive")
            .default_value(false)
            .implicit_value(true)
            .help("Recursively scan through directories to load FAST5 and POD5 files");

    parser.add_argument("--modified-bases")
            .nargs(argparse::nargs_pattern::at_least_one)
            .action([](const std::string& value) {
                if (std::find(modified::mods.begin(), modified::mods.end(), value) ==
                    modified::mods.end()) {
                    spdlog::error(
                            "'{}' is not a supported modification please select from {}", value,
                            std::accumulate(std::next(modified::mods.begin()), modified::mods.end(),
                                            modified::mods[0], [](std::string a, std::string b) {
                                                return a + ", " + b;
                                            }));
                    std::exit(EXIT_FAILURE);
                }
                return value;
            });

    parser.add_argument("--modified-bases-models")
            .default_value(std::string())
            .help("a comma separated list of modified base models");

    parser.add_argument("--modified-bases-threshold")
            .default_value(default_parameters.methylation_threshold)
            .scan<'f', float>()
            .help("the minimum predicted methylation probability for a modified base to be emitted "
                  "in an all-context model, [0, 1]");

    parser.add_argument("--emit-fastq")
            .help("Output in fastq format.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--emit-sam")
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--emit-moves").default_value(false).implicit_value(true);

    parser.add_argument("--reference")
            .help("Path to reference for alignment.")
            .default_value(std::string(""));
    parser.add_argument("-k")
            .help("k-mer size for alignment with minimap2 (maximum 28).")
            .default_value(15)
            .scan<'i', int>();
    parser.add_argument("-w")
            .help("minimizer window size for alignment with minimap2.")
            .default_value(10)
            .scan<'i', int>();
    parser.add_argument("-I").help("minimap2 index batch size.").default_value(std::string("16G"));

    argparse::ArgumentParser internal_parser;

    try {
        auto remaining_args = parser.parse_known_args(argc, argv);
        internal_parser = utils::parse_internal_options(remaining_args);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto model = parser.get<std::string>("model");
    auto mod_bases = parser.get<std::vector<std::string>>("--modified-bases");
    auto mod_bases_models = parser.get<std::string>("--modified-bases-models");

    if (mod_bases.size() && !mod_bases_models.empty()) {
        spdlog::error(
                "only one of --modified-bases or --modified-bases-models should be specified.");
        std::exit(EXIT_FAILURE);
    } else if (mod_bases.size()) {
        std::vector<std::string> m;
        std::transform(mod_bases.begin(), mod_bases.end(), std::back_inserter(m),
                       [&model](std::string m) { return utils::get_modification_model(model, m); });

        mod_bases_models =
                std::accumulate(std::next(m.begin()), m.end(), m[0],
                                [](std::string a, std::string b) { return a + "," + b; });
    }

    auto methylation_threshold = parser.get<float>("--modified-bases-threshold");
    if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
        spdlog::error("--modified-bases-threshold must be between 0 and 1.");
        std::exit(EXIT_FAILURE);
    }

    auto output_mode = HtsWriter::OutputMode::BAM;

    auto emit_fastq = parser.get<bool>("--emit-fastq");
    auto emit_sam = parser.get<bool>("--emit-sam");

    if (emit_fastq && emit_sam) {
        throw std::runtime_error("Only one of --emit-{fastq, sam} can be set (or none).");
    }

    if (emit_fastq) {
        output_mode = HtsWriter::OutputMode::FASTQ;
    } else if (emit_sam || utils::is_fd_tty(stdout)) {
        output_mode = HtsWriter::OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = HtsWriter::OutputMode::UBAM;
    }

    spdlog::info("> Creating basecall pipeline");

    try {
        setup(args, model, parser.get<std::string>("data"), mod_bases_models,
              parser.get<std::string>("-x"), parser.get<std::string>("--reference"),
              parser.get<int>("-c"), parser.get<int>("-o"), parser.get<int>("-b"),
              default_parameters.num_runners, default_parameters.remora_batchsize,
              default_parameters.remora_threads, methylation_threshold, output_mode,
              parser.get<bool>("--emit-moves"), parser.get<int>("--max-reads"),
              parser.get<int>("--min-qscore"), parser.get<std::string>("--read-ids"),
              parser.get<bool>("--recursive"), parser.get<int>("k"), parser.get<int>("w"),
              utils::parse_string_to_size(parser.get<std::string>("I")),
              internal_parser.get<bool>("--skip-model-compatibility-check"));
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }

    spdlog::info("> Finished");
    return 0;
}

}  // namespace dorado
