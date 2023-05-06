#include "Version.h"
#include "utils/bam_utils.h"
#include "utils/log_utils.h"

#include <argparse.hpp>
#include <spdlog/spdlog.h>

#include <csignal>
#include <filesystem>

namespace dorado {

volatile sig_atomic_t interupt = 0;

using HtsReader = utils::HtsReader;

std::vector<std::string> header = {
        "filename",
        "read_id",
        "run_id",
        "channel",
        "mux",
        "start_time",
        "duration",
        //"template_start",
        "template_duration",
        "sequence_length_template",
        "mean_qscore_template",
};

int summary(int argc, char* argv[]) {
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("reads").help("SAM/BAM file produced by dorado basecaller.");
    parser.add_argument("-s", "--separator").default_value(std::string("\t"));
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        std::exit(1);
    }

    if (parser.get<bool>("--verbose")) {
        spdlog::set_level(spdlog::level::debug);
    }

    auto reads(parser.get<std::string>("reads"));
    auto separator(parser.get<std::string>("separator"));

    HtsReader reader(reads);
    spdlog::debug("> input fmt: {} aligned: {}", reader.format, reader.is_aligned);
#ifndef _WIN32
    std::signal(SIGPIPE, [](int signum) { interupt = 1; });
#endif
    std::signal(SIGINT, [](int signum) { interupt = 1; });

    for (int col = 0; col < header.size(); col++) {
        std::cout << header[col] << separator;
    }
    std::cout << header[header.size()] << '\n';

    while (reader.read() && !interupt) {
        auto rg_value = reader.get_tag<std::string>("RG");
        auto filename = reader.get_tag<std::string>("f5");
        if (filename.empty()) {
            filename = reader.get_tag<std::string>("fn");
        }
        auto read_id = bam_get_qname(reader.record);
        auto channel = reader.get_tag<int>("ch");
        auto mux = reader.get_tag<int>("mx");
        auto start_time = reader.get_tag<std::string>("st");
        auto duration = reader.get_tag<float>("du");
        auto seqlen = reader.record->core.l_qseq;
        auto mean_qscore = reader.get_tag<int>("qs");

        auto num_samples = reader.get_tag<int>("ns");
        auto trim_samples = reader.get_tag<int>("ts");
        // todo: sample_rate
        float template_duration = (num_samples - trim_samples) / 4000.0f;

        std::cout << filename << separator << read_id << separator << rg_value.substr(0, 36)
                  << separator << channel << separator << mux << separator << start_time
                  << separator << duration << separator << template_duration << separator << seqlen
                  << separator << mean_qscore << separator << '\n';
    }

    return 0;
}

}  // namespace dorado
