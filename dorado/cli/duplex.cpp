#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#include "nn/CudaCRFModel.h"
#include "nn/MetalCRFModel.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/cuda_utils.h"
#include "utils/duplex_utils.h"
#include "utils/log_utils.h"
#ifdef __APPLE__
#include "utils/metal_utils.h"
#endif

#include <argparse.hpp>
#include <spdlog/spdlog.h>
#include "utils/parameters.h"
#include <thread>

namespace dorado {

int duplex(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION);
    parser.add_argument("model").help("Model");
    parser.add_argument("reads").help("Reads in Pod5 format or BAM/SAM format for basespace.");
    parser.add_argument("--pairs").help("Space-delimited csv containing read ID pairs.");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);
    parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();
    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
            .default_value(utils::default_parameters.device);
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        std::exit(1);
    }

    std::string device = parser.get<std::string>("-x");

    std::string model = parser.get<std::string>("model");
    std::string reads = parser.get<std::string>("reads");
    std::string pairs_file = parser.get<std::string>("--pairs");
    size_t threads = static_cast<size_t>(parser.get<int>("--threads"));
    bool emit_fastq = parser.get<bool>("--emit-fastq");
    std::vector<std::string> args(argv, argv + argc);

    spdlog::info("> Loading pairs file");
    std::map<std::string, std::string> template_complement_map = utils::load_pairs_file(pairs_file);

    WriterNode writer_node(std::move(args), emit_fastq, false, true, 4);
    torch::set_num_threads(1);

    if (model.compare("basespace") == 0) {
        spdlog::info("> Loading reads");
        std::map<std::string, std::shared_ptr<Read>> read_map = utils::read_bam(reads);

        threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
        BaseSpaceDuplexCallerNode duplex_caller_node(writer_node, template_complement_map, read_map,
                                                     threads);
    } else {  // Execute a Stereo Basecall pipeline.

        // ***** STEREO STARTS HERE *** //
        torch::set_num_threads(1);
        std::vector<Runner> runners;
        int num_devices = 1;
        size_t batch_size;
        size_t model_stride = 5;   // TODO: Set in CLI
        size_t chunk_size = 5000;  // TODO: Set in CLI
        size_t overlap = 100;      // TODO: Set in CLI
        size_t num_runners = 1;


        if (device == "cpu") {
            batch_size = batch_size == 0 ? std::thread::hardware_concurrency() : batch_size;
            for (size_t i = 0; i < num_runners; i++) {
                runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(
                        model, device, chunk_size, batch_size));
            }

#ifdef __APPLE__
        } else if (device == "metal") {
            batch_size = batch_size == 0 ? utils::auto_gpu_batch_size(model) : batch_size;
            auto caller = create_metal_caller(model, chunk_size, batch_size);
            for (int i = 0; i < num_runners; i++) {
                runners.push_back(
                        std::make_shared<MetalModelRunner>(caller, chunk_size, batch_size));
            }
        } else {
            throw std::runtime_error(std::string("Unsupported device: ") + device);
        }
#else   // ifdef __APPLE__
        } else {
            auto devices = utils::parse_cuda_device_string(device);
            num_devices = devices.size();
            if (num_devices == 0) {
                throw std::runtime_error("CUDA device requested but no devices found.");
            }
            batch_size = batch_size == 0 ? utils::auto_gpu_batch_size(model, devices)
                                         : batch_size;
            for (auto device_string : devices) {
                auto caller = create_cuda_caller(model, chunk_size, batch_size, device_string);
                for (size_t i = 0; i < num_runners; i++) {
                    runners.push_back(
                            std::make_shared<CudaModelRunner>(caller, chunk_size, batch_size));
                }
            }
        }
#endif  // __APPLE__

        StereoDuplexEncoderNode stereo_node = StereoDuplexEncoderNode(
                writer_node,
                std::move(
                        template_complement_map));  //Currently the StereoDuplexEncoderNode just outputs the read it receives into the writer node

        std::unique_ptr<BasecallerNode> basecaller_node;
        basecaller_node = std::make_unique<BasecallerNode>(
                stereo_node, std::move(runners), batch_size, chunk_size, overlap, model_stride);
        ScalerNode scaler_node(*basecaller_node, num_devices * 2);
        DataLoader loader(scaler_node, "cpu", num_devices);
        loader.load_reads(reads);
    }

    return 0;
}
}  // namespace dorado
