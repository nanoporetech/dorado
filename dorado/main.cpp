#include <iostream>
#include <argparse.hpp>

#include "decode/Decoder.h"
#include "nn/ModelRunnerCPU.h"
#include "nn/ModelRunnerGPU.h"
#include "data_loader/Fast5DataLoader.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/WriterNode.h"


void setup(std::string model, std::string data, std::string device, size_t chunk_size, size_t overlap, size_t batch_size, size_t num_runners) {

    auto decode_options = DecoderOptions();
    std::vector<std::shared_ptr<ModelRunner>> runners;

    if (device == "cpu") {
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunnerCPU>(model, device, chunk_size, batch_size, decode_options));
        }
    } else {
        for (int i = 0; i < num_runners; i++) {
            runners.push_back(std::make_shared<ModelRunnerGPU>(model, device, chunk_size, batch_size, decode_options));
        }
    }

    WriterNode writer_node;
    BasecallerNode basecaller_node(writer_node, runners, batch_size, chunk_size, overlap);
    ScalerNode scaler_node(basecaller_node);
    Fast5DataLoader loader(scaler_node, "cpu");
    loader.load_reads(data);
}


int main(int argc, char *argv[]) {

    argparse::ArgumentParser parser("dorado", "0.0.1a0");

    parser.add_argument("model")
            .help("the basecaller model to run.");

    parser.add_argument("data")
            .help("the data directory.");

    parser.add_argument("-x", "--device")
#ifdef __APPLE__
            .default_value(std::string{"metal"});
#else
            .default_value(std::string{"cuda:0"});
#endif

    parser.add_argument("-b", "--batchsize")
            .default_value(1024)
            .scan<'i', int>();

    parser.add_argument("-c", "--chunksize")
            .default_value(8000)
            .scan<'i', int>();

    parser.add_argument("-o", "--overlap")
            .default_value(150)
            .scan<'i', int>();

    parser.add_argument("-r", "--num_runners")
            .default_value(1)
            .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    std::cerr << "> Creating basecall pipeline" << std::endl;
    try {
        setup(
            parser.get<std::string>("model"),
            parser.get<std::string>("data"),
            parser.get<std::string>("-x"),
            parser.get<int>("-c"),
            parser.get<int>("-o"),
            parser.get<int>("-b"),
            parser.get<int>("-r")
        );
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::exit(1);
    }

    std::cerr << "> Finished" << std::endl;

}
