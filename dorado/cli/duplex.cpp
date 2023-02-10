#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "read_pipeline/WriterNode.h"
#include "utils/bam_utils.h"
#include "utils/duplex_utils.h"
#include "utils/log_utils.h"
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#include "utils/metal_utils.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif

#include "utils/parameters.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <thread>

namespace dorado {

int duplex(int argc, char* argv[]) {
    using dorado::utils::default_parameters;
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

    parser.add_argument("-r", "--num_runners")
            .default_value(default_parameters.num_runners)
            .scan<'i', int>();

    parser.add_argument("--min-qscore").default_value(0).scan<'i', int>();

    try {
        parser.parse_args(argc, argv);

        auto device(parser.get<std::string>("-x"));
        auto model(parser.get<std::string>("model"));
        auto reads(parser.get<std::string>("reads"));
        auto pairs_file(parser.get<std::string>("--pairs"));
        auto threads = static_cast<size_t>(parser.get<int>("--threads"));
        bool emit_fastq = parser.get<bool>("--emit-fastq");
        auto min_qscore(parser.get<int>("--min-qscore"));
        std::vector<std::string> args(argv, argv + argc);

        spdlog::info("> Loading pairs file");
        std::map<std::string, std::string> template_complement_map =
                utils::load_pairs_file(pairs_file);
        spdlog::info("> Pairs file loaded");

        bool emit_moves = false, rna = false, duplex = true;
        WriterNode writer_node(std::move(args), emit_fastq, emit_moves, rna, duplex, min_qscore, 4);

        torch::set_num_threads(1);

        if (model.compare("basespace") == 0) {  // Execute a Basespace duplex pipeline.
            spdlog::info("> Loading reads");
            std::map<std::string, std::shared_ptr<Read>> read_map = utils::read_bam(reads);
            spdlog::info("> Starting Basespace Duplex Pipeline");

            threads = threads == 0 ? std::thread::hardware_concurrency() : threads;
            BaseSpaceDuplexCallerNode duplex_caller_node(writer_node, template_complement_map,
                                                         read_map, threads);
        } else {  // Execute a Stereo Duplex pipeline.
            torch::set_num_threads(1);
            std::vector<Runner> runners;
            std::vector<Runner> stereo_runners;

            int num_devices;
            int batch_size(parser.get<int>("-b"));
            int chunk_size(parser.get<int>("-c"));
            int overlap(parser.get<int>("-o"));
            size_t num_runners = 1;

            size_t stereo_batch_size;

            if (device == "cpu") {
                batch_size = batch_size == 0 ? std::thread::hardware_concurrency() : batch_size;
                for (size_t i = 0; i < num_runners; i++) {
                    runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(
                            model, device, chunk_size, batch_size));
                }
#ifdef __APPLE__
            } else {
                throw std::runtime_error(
                        std::string("Stereo Duplex currently unspported on Metal"));
            }
#else   // ifdef __APPLE__
            } else {
                auto devices = utils::parse_cuda_device_string(device);
                num_devices = devices.size();
                if (num_devices == 0) {
                    throw std::runtime_error("CUDA device requested but no devices found.");
                }
                batch_size =
                        batch_size == 0 ? utils::auto_gpu_batch_size(model, devices) : batch_size;
                batch_size = batch_size / 2;  //Needed to support Stereo and Simplex in parallel
                for (auto device_string : devices) {
                    auto caller = create_cuda_caller(model, chunk_size, batch_size, device_string);
                    for (size_t i = 0; i < num_runners; i++) {
                        runners.push_back(
                                std::make_shared<CudaModelRunner>(caller, chunk_size, batch_size));
                    }
                }

                stereo_batch_size = 1024;

                std::string stereo_model_name("dna_r10.4.2_e8.2_4khz_stereo@v1.0");
                auto model_path = std::filesystem::path(model);
                auto stereo_model_path =
                        model_path.parent_path() / std::filesystem::path(stereo_model_name);

                std::string stereo_model(stereo_model_path.string());

                for (auto device_string : devices) {
                    auto caller = create_cuda_caller(stereo_model, chunk_size, stereo_batch_size,
                                                     device_string);
                    for (size_t i = 0; i < num_runners; i++) {
                        stereo_runners.push_back(std::make_shared<CudaModelRunner>(
                                caller, chunk_size, stereo_batch_size));
                    }
                }
            }
#endif  // __APPLE__
            spdlog::info("> Starting Stereo Duplex pipeline");

            std::unique_ptr<BasecallerNode> stereo_basecaller_node;

            auto stereo_model_stride = stereo_runners.front()->model_stride();
            stereo_basecaller_node = std::make_unique<BasecallerNode>(
                    writer_node, std::move(stereo_runners), stereo_batch_size, chunk_size, overlap,
                    stereo_model_stride);

            StereoDuplexEncoderNode stereo_node = StereoDuplexEncoderNode(
                    *stereo_basecaller_node, std::move(template_complement_map));

            std::unique_ptr<BasecallerNode> basecaller_node;
            auto simplex_model_stride = runners.front()->model_stride();
            basecaller_node =
                    std::make_unique<BasecallerNode>(stereo_node, std::move(runners), batch_size,
                                                     chunk_size, overlap, simplex_model_stride);
            ScalerNode scaler_node(*basecaller_node, num_devices * 2);

            DataLoader loader(scaler_node, "cpu", num_devices);
            loader.load_reads(reads);
        }
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        std::exit(1);
    }
    return 0;
}
}  // namespace dorado
