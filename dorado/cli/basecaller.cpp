#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#include "utils/basecaller_utils.h"
#include "utils/models.h"
#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#include "utils/metal_utils.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD
#include "nn/ModelRunner.h"
#include "nn/RemoraModel.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

namespace dorado {

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
           bool emit_fastq,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           std::string read_list_file_path,
           bool recursive_file_loading) {
    torch::set_num_threads(1);
    std::vector<Runner> runners;

    // Default is 1 device.  CUDA path may alter this.
    int num_devices = 1;

    if (device == "cpu") {
        batch_size = batch_size == 0 ? std::thread::hardware_concurrency() : batch_size;
        for (size_t i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(model_path, device,
                                                                        chunk_size, batch_size));
        }
    }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
    else if (device == "metal") {
        if (batch_size == 0) {
            batch_size = utils::auto_gpu_batch_size();
            spdlog::debug("- selected batchsize {}", batch_size);
        }
        auto caller = create_metal_caller(model_path, chunk_size, batch_size);
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<MetalModelRunner>(caller, chunk_size, batch_size));
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
        batch_size = batch_size == 0 ? utils::auto_gpu_batch_size(model_path.string(), devices)
                                     : batch_size;

        spdlog::debug("- selected batchsize {}", batch_size);

        for (auto device_string : devices) {
            auto caller = create_cuda_caller(model_path, chunk_size, batch_size, device_string);
            for (size_t i = 0; i < num_runners; i++) {
                runners.push_back(
                        std::make_shared<CudaModelRunner>(caller, chunk_size, batch_size));
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

    if (!remora_models.empty() && emit_fastq) {
        throw std::runtime_error("Modified base models cannot be used with FASTQ output");
    }

    std::vector<std::filesystem::path> remora_model_list;
    std::istringstream stream{remora_models};
    std::string model;
    while (std::getline(stream, model, ',')) {
        remora_model_list.push_back(model);
    }

    // generate model callers before nodes or it affects the speed calculations
    std::vector<std::shared_ptr<RemoraCaller>> remora_callers;

#if DORADO_GPU_BUILD && !defined(__APPLE__)
    if (device != "cpu") {
        auto devices = utils::parse_cuda_device_string(device);
        num_devices = devices.size();

        for (auto device_string : devices) {
            for (const auto& remora_model : remora_model_list) {
                auto caller = std::make_shared<RemoraCaller>(remora_model, device_string,
                                                             remora_batch_size, model_stride);
                remora_callers.push_back(caller);
            }
        }
    } else
#endif
    {
        for (const auto& remora_model : remora_model_list) {
            auto caller = std::make_shared<RemoraCaller>(remora_model, device, remora_batch_size,
                                                         model_stride);
            remora_callers.push_back(caller);
        }
    }

    std::string model_name = std::filesystem::canonical(model_path).filename().string();
    auto read_groups = DataLoader::load_read_groups(data_path, model_name, recursive_file_loading);

    auto read_list = utils::load_read_list(read_list_file_path);

    size_t num_reads = DataLoader::get_num_reads(data_path, read_list, recursive_file_loading);
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    bool rna = utils::is_rna_model(model_path), duplex = false;

    std::unique_ptr<utils::BamWriter> bam_writer;
    std::shared_ptr<utils::Aligner> aligner;
    std::shared_ptr<WriterNode> writer_node;
    MessageSink* filter_sink = nullptr;
    if (ref.empty()) {
        writer_node =
                std::make_shared<WriterNode>(std::move(args), emit_fastq, emit_moves, rna, duplex,
                                             num_devices * 2, std::move(read_groups), num_reads);
        filter_sink = writer_node.get();
    } else {
        bam_writer = std::make_unique<utils::BamWriter>("-", num_devices /*writer_threads*/);
        aligner = std::make_shared<utils::Aligner>(*bam_writer, ref, 19, 19, num_devices * 5);
        bam_writer->write_header(nullptr, aligner->sq());
        filter_sink = aligner.get();
    }
    ReadToBamType read_converter(*filter_sink, emit_moves, rna, duplex, 1 /*num_threads*/);
    ReadFilterNode read_filter_node(read_converter, min_qscore, 1, num_reads);

    std::unique_ptr<ModBaseCallerNode> mod_base_caller_node;
    std::unique_ptr<BasecallerNode> basecaller_node;

    const int kBatchTimeoutMS = 100;
    if (!remora_model_list.empty()) {
        mod_base_caller_node = std::make_unique<ModBaseCallerNode>(
                read_filter_node, std::move(remora_callers), num_remora_threads, num_devices,
                model_stride, remora_batch_size);
        basecaller_node = std::make_unique<BasecallerNode>(
                *mod_base_caller_node, std::move(runners), batch_size, chunk_size, overlap,
                model_stride, kBatchTimeoutMS, model_name);
    } else {
        basecaller_node = std::make_unique<BasecallerNode>(
                read_filter_node, std::move(runners), batch_size, chunk_size, overlap, model_stride,
                kBatchTimeoutMS, model_name);
    }
    ScalerNode scaler_node(*basecaller_node, num_devices * 2);
    DataLoader loader(scaler_node, "cpu", num_devices, max_reads, read_list);

    loader.load_reads(data_path, recursive_file_loading);
}

int basecaller(int argc, char* argv[]) {
    using dorado::utils::default_parameters;

    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("model").help("the basecaller model to run.");

    parser.add_argument("data").help("the data directory.");

    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
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

    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    parser.add_argument("--emit-moves").default_value(false).implicit_value(true);

    parser.add_argument("--ref").help("Path to reference for alignment.");

    try {
        parser.parse_args(argc, argv);
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

    spdlog::info("> Creating basecall pipeline");

    try {
        setup(args, model, parser.get<std::string>("data"), mod_bases_models,
              parser.get<std::string>("-x"), parser.get<std::string>("--ref"),
              parser.get<int>("-c"), parser.get<int>("-o"), parser.get<int>("-b"),
              default_parameters.num_runners, default_parameters.remora_batchsize,
              default_parameters.remora_threads, parser.get<bool>("--emit-fastq"),
              parser.get<bool>("--emit-moves"), parser.get<int>("--max-reads"),
              parser.get<int>("--min-qscore"), parser.get<std::string>("--read-ids"),
              parser.get<bool>("--recursive"));
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }

    spdlog::info("> Finished");
    return 0;
}

}  // namespace dorado
