#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/parse_cuda_device_string.h"

#endif
#include "nn/ModelRunner.h"
#include "nn/RemoraModel.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ModBaseCallerNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/WriterNode.h"

#include <argparse.hpp>

#include <filesystem>
#include <iostream>
#include <sstream>
#include <thread>

void setup(std::vector<std::string> args,
           const std::filesystem::path& model_path,
           const std::string& data_path,
           const std::string& remora_models,
           const std::string& device,
           size_t chunk_size,
           size_t overlap,
           size_t batch_size,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           bool emit_fastq) {
    torch::set_num_threads(1);
    std::vector<Runner> runners;

    int num_devices = 1;

    if (device == "cpu") {
        batch_size = batch_size == 0 ? std::thread::hardware_concurrency() : batch_size;
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(model_path, device,
                                                                        chunk_size, batch_size));
        }
#ifdef __APPLE__
    } else if (device == "metal") {
        batch_size = batch_size == 0 ? 384 : batch_size;
        auto caller = create_metal_caller(model_path, chunk_size, batch_size);
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<MetalModelRunner>(caller, chunk_size, batch_size));
        }
    } else {
        throw std::runtime_error(std::string("Unsupported device: ") + device);
    }
#else   // ifdef __APPLE__
    } else {
        auto devices = parse_cuda_device_string(device);
        num_devices = devices.size();
        batch_size =
                batch_size == 0 ? auto_gpu_batch_size(model_path.string(), devices) : batch_size;
        for (auto device_string : devices) {
            auto caller = create_cuda_caller(model_path, chunk_size, batch_size, device_string);
            for (int i = 0; i < num_runners; i++) {
                runners.push_back(
                        std::make_shared<CudaModelRunner>(caller, chunk_size, batch_size));
            }
        }
    }
#endif  // __APPLE__

    // verify that all runners are using the same stride, in case we allow multiple models in future
    auto model_stride = runners.front()->model_stride();
    assert(std::all_of(runners.begin(), runners.end(), [model_stride](auto runner) {
        return runner->model_stride() == model_stride;
    }));

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
    for (const auto& remora_model : remora_model_list) {
        auto caller = std::make_shared<RemoraCaller>(remora_model, device, remora_batch_size,
                                                     model_stride);
        remora_callers.push_back(caller);
    }

    WriterNode writer_node(std::move(args), emit_fastq);

    std::unique_ptr<ModBaseCallerNode> mod_base_caller_node;
    std::unique_ptr<BasecallerNode> basecaller_node;

    if (!remora_model_list.empty()) {
        mod_base_caller_node.reset(new ModBaseCallerNode(writer_node, std::move(remora_callers),
                                                         num_remora_threads, model_stride,
                                                         remora_batch_size));
        basecaller_node =
                std::make_unique<BasecallerNode>(*mod_base_caller_node, std::move(runners),
                                                 batch_size, chunk_size, overlap, model_stride);
    } else {
        basecaller_node = std::make_unique<BasecallerNode>(
                writer_node, std::move(runners), batch_size, chunk_size, overlap, model_stride);
    }
    ScalerNode scaler_node(*basecaller_node, num_devices * 5);
    DataLoader loader(scaler_node, "cpu");
    loader.load_reads(data_path);
}

int basecaller(int argc, char* argv[]) {
    argparse::ArgumentParser parser("dorado", DORADO_VERSION);

    parser.add_argument("model").help("the basecaller model to run.");

    parser.add_argument("data").help("the data directory.");

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
#ifdef __APPLE__
            .default_value(std::string{"metal"});
#else
            .default_value(std::string{"cuda:all"});
#endif

    parser.add_argument("-b", "--batchsize").default_value(0).scan<'i', int>();

    parser.add_argument("-c", "--chunksize").default_value(10000).scan<'i', int>();

    parser.add_argument("-o", "--overlap").default_value(500).scan<'i', int>();

    parser.add_argument("-r", "--num_runners").default_value(2).scan<'i', int>();

    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);

    parser.add_argument("--remora-batchsize").default_value(1000).scan<'i', int>();

    parser.add_argument("--remora-threads").default_value(1).scan<'i', int>();

    parser.add_argument("--remora_models")
            .default_value(std::string())
            .help("a comma separated list of remora models");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    std::cerr << "> Creating basecall pipeline" << std::endl;
    try {
        setup(args, parser.get<std::string>("model"), parser.get<std::string>("data"),
              parser.get<std::string>("--remora_models"), parser.get<std::string>("-x"),
              parser.get<int>("-c"), parser.get<int>("-o"), parser.get<int>("-b"),
              parser.get<int>("-r"), parser.get<int>("--remora-batchsize"),
              parser.get<int>("--remora-threads"), parser.get<bool>("--emit-fastq"));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    std::cerr << "> Finished" << std::endl;
    return 0;
}
