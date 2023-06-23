#include "Version.h"
#include "data_loader/DataLoader.h"
#include "decode/CPUDecoder.h"
#include "nn/CRFModel.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/BasecallerNode.h"
#include "read_pipeline/DuplexSplitNode.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/PairingNode.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ScalerNode.h"
#include "read_pipeline/StereoDuplexEncoderNode.h"
#include "utils/bam_utils.h"
#include "utils/cli_utils.h"
#include "utils/duplex_utils.h"
#include "utils/log_utils.h"
#if DORADO_GPU_BUILD
#ifdef __APPLE__
#include "nn/MetalCRFModel.h"
#else
#include "nn/CudaCRFModel.h"
#include "utils/cuda_utils.h"
#endif
#endif  // DORADO_GPU_BUILD

#include "utils/models.h"
#include "utils/parameters.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <utils/basecaller_utils.h>

#include <memory>
#include <thread>
#include <unordered_set>

namespace dorado {

using dorado::utils::default_parameters;
using namespace std::chrono_literals;

int duplex(int argc, char* argv[]) {
    using dorado::utils::default_parameters;
    utils::InitLogging();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    parser.add_argument("model").help("Model");
    parser.add_argument("reads").help("Reads in Pod5 format or BAM/SAM format for basespace.");
    parser.add_argument("--pairs")
            .default_value(std::string(""))
            .help("Space-delimited csv containing read ID pairs. If not provided, pairing will be "
                  "performed automatically");
    parser.add_argument("--emit-fastq").default_value(false).implicit_value(true);
    parser.add_argument("--emit-sam")
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();

    parser.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
            .default_value(utils::default_parameters.device);

    parser.add_argument("-b", "--batchsize")
            .default_value(default_parameters.batchsize)
            .scan<'i', int>()
            .help("if 0 an optimal batchsize will be selected. batchsizes are rounded to the "
                  "closest multiple of 64.");

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

    parser.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to basecall. If not provided, all "
                  "reads will be basecalled")
            .default_value(std::string(""));

    parser.add_argument("--min-qscore").default_value(0).scan<'i', int>();

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
    parser.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    parser.add_argument("-I").help("minimap2 index batch size.").default_value(std::string("16G"));

    try {
        auto remaining_args = parser.parse_known_args(argc, argv);
        auto internal_parser = utils::parse_internal_options(remaining_args);

        auto device(parser.get<std::string>("-x"));
        auto model(parser.get<std::string>("model"));

        if (model.find("fast") != std::string::npos) {
            spdlog::warn("Fast models are currently not recommended for duplex basecalling.");
        }

        auto reads(parser.get<std::string>("reads"));
        std::string pairs_file = parser.get<std::string>("--pairs");
        auto threads = static_cast<size_t>(parser.get<int>("--threads"));
        auto min_qscore(parser.get<int>("--min-qscore"));
        auto ref = parser.get<std::string>("--reference");
        const bool basespace_duplex = (model.compare("basespace") == 0);
        std::vector<std::string> args(argv, argv + argc);
        if (parser.get<bool>("--verbose")) {
            utils::SetDebugLogging();
        }
        std::map<std::string, std::string> template_complement_map;
        auto read_list = utils::load_read_list(parser.get<std::string>("--read-ids"));

        std::unordered_set<std::string> read_list_from_pairs;

        if (!pairs_file.empty()) {
            spdlog::info("> Loading pairs file");
            template_complement_map = utils::load_pairs_file(pairs_file);
            read_list_from_pairs = utils::get_read_list_from_pairs(template_complement_map);
            spdlog::info("> Pairs file loaded with {} reads.", read_list_from_pairs.size());
        } else {
            spdlog::info(
                    "> No duplex pairs file provided, pairing will be performed automatically");
        }

        bool emit_moves = false, rna = false, duplex = true;

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

        bool recursive_file_loading = parser.get<bool>("--recursive");

        size_t num_reads = (basespace_duplex ? read_list_from_pairs.size()
                                             : DataLoader::get_num_reads(reads, read_list, {},
                                                                         recursive_file_loading));
        spdlog::debug("> Reads to process: {}", num_reads);

        std::unique_ptr<sam_hdr_t, void (*)(sam_hdr_t*)> hdr(sam_hdr_init(), sam_hdr_destroy);
        utils::add_pg_hdr(hdr.get(), args);
        std::shared_ptr<HtsWriter> bam_writer;
        std::shared_ptr<Aligner> aligner;
        MessageSink* converted_reads_sink = nullptr;

        if (ref.empty()) {
            bam_writer = std::make_shared<HtsWriter>("/dev/null", output_mode, 4, num_reads);
            converted_reads_sink = bam_writer.get();
        } else {
            bam_writer = std::make_shared<HtsWriter>("-", output_mode, 4, num_reads);
            aligner = std::make_shared<Aligner>(
                    *bam_writer, ref, parser.get<int>("k"), parser.get<int>("w"),
                    utils::parse_string_to_size(parser.get<std::string>("I")),
                    std::thread::hardware_concurrency());
            utils::add_sq_hdr(hdr.get(), aligner->get_sequence_records_for_header());
            converted_reads_sink = aligner.get();
        }
        ReadToBamType read_converter(*converted_reads_sink, emit_moves, rna, 2);
        // The minimum sequence length is set to 5 to avoid issues with duplex node printing very short sequences for mismatched pairs.
        ReadFilterNode read_filter_node(read_converter, min_qscore,
                                        default_parameters.min_seqeuence_length, {}, 5);

        torch::set_num_threads(1);

        // Add stats reporting for shared nodes.
        using dorado::stats::make_stats_reporter;
        std::vector<dorado::stats::StatsReporter> stats_reporters;
        if (aligner) {
            stats_reporters.push_back(make_stats_reporter(*aligner));
        }
        stats_reporters.push_back(make_stats_reporter(*bam_writer));
        stats_reporters.push_back(make_stats_reporter(read_filter_node));

        std::vector<dorado::stats::StatsCallable> stats_callables;
        ProgressTracker tracker(num_reads, duplex);
        stats_callables.push_back(
                [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });

        if (basespace_duplex) {  // Execute a Basespace duplex pipeline.
            if (pairs_file.empty()) {
                spdlog::error("The --pairs argument is required for the basespace model.");
                return 1;  // Exit with an error code
            }
            // Write header as no read group info is needed.
            bam_writer->write_header(hdr.get());

            spdlog::info("> Loading reads");
            auto read_map = read_bam(reads, read_list_from_pairs);

            spdlog::info("> Starting Basespace Duplex Pipeline");
            threads = threads == 0 ? std::thread::hardware_concurrency() : threads;

            constexpr auto kStatsPeriod = 100ms;
            auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables);
            // End stats counting setup.

            BaseSpaceDuplexCallerNode duplex_caller_node(read_filter_node, template_complement_map,
                                                         read_map, threads);
            bam_writer->join();  // Explicitly wait for all output rows to be written.
            stats_sampler->terminate();
        } else {  // Execute a Stereo Duplex pipeline.

            const auto model_path = std::filesystem::canonical(std::filesystem::path(model));
            model = model_path.filename().string();
            auto model_config = load_crf_model_config(model_path);

            auto data_sample_rate = DataLoader::get_sample_rate(reads, recursive_file_loading);
            auto model_sample_rate = get_model_sample_rate(model_path);
            auto skip_model_compatibility_check =
                    internal_parser.get<bool>("--skip-model-compatibility-check");
            if (!skip_model_compatibility_check && (data_sample_rate != model_sample_rate)) {
                std::stringstream err;
                err << "Sample rate for model (" << model_sample_rate << ") and data ("
                    << data_sample_rate << ") don't match.";
                throw std::runtime_error(err.str());
            }
            auto stereo_model_name = utils::get_stereo_model_name(model, data_sample_rate);
            const auto stereo_model_path =
                    model_path.parent_path() / std::filesystem::path(stereo_model_name);

            if (!std::filesystem::exists(stereo_model_path)) {
                utils::download_models(model_path.parent_path().u8string(), stereo_model_name);
            }
            auto stereo_model_config = load_crf_model_config(stereo_model_path);

            // Write read group info to header.
            auto read_groups = DataLoader::load_read_groups(reads, model, recursive_file_loading);
            auto duplex_rg_name = std::string(model + "_" + stereo_model_name);
            read_groups.merge(
                    DataLoader::load_read_groups(reads, duplex_rg_name, recursive_file_loading));
            utils::add_rg_hdr(hdr.get(), read_groups);
            bam_writer->write_header(hdr.get());

            std::vector<Runner> runners;
            std::vector<Runner> stereo_runners;

            // Default is 1 device.  CUDA path may alter this.
            int num_devices = 1;
            int batch_size(parser.get<int>("-b"));
            int chunk_size(parser.get<int>("-c"));
            int overlap(parser.get<int>("-o"));
            const size_t num_runners = default_parameters.num_runners;

            if (device == "cpu") {
                if (batch_size == 0) {
                    batch_size = std::thread::hardware_concurrency();
                    spdlog::debug("- set batch size to {}", batch_size);
                }
                for (size_t i = 0; i < num_runners; i++) {
                    runners.push_back(std::make_shared<ModelRunner<CPUDecoder>>(
                            model_path, device, chunk_size, batch_size));
                }
            }
#if DORADO_GPU_BUILD
#ifdef __APPLE__
            else if (device == "metal") {
                auto simplex_caller =
                        create_metal_caller(model_config, model_path, chunk_size, batch_size);
                for (int i = 0; i < num_runners; i++) {
                    runners.push_back(std::make_shared<MetalModelRunner>(simplex_caller));
                }
                if (runners.back()->batch_size() != batch_size) {
                    spdlog::debug("- set batch size to {}", runners.back()->batch_size());
                }

                // For now, the minimal batch size is used for the duplex model.
                int stereo_batch_size = 48;

                auto duplex_caller = create_metal_caller(stereo_model_config, stereo_model_path,
                                                         chunk_size, stereo_batch_size);
                for (size_t i = 0; i < num_runners; i++) {
                    stereo_runners.push_back(std::make_shared<MetalModelRunner>(duplex_caller));
                }
            } else {
                throw std::runtime_error(std::string("Unsupported device: ") + device);
            }
#else   // ifdef __APPLE__
            else {
                // Note: The memory assignment between simplex and duplex callers have been
                // performed based on empirical results considering a SUP model for simplex
                // calling.
                auto devices = utils::parse_cuda_device_string(device);
                num_devices = devices.size();
                if (num_devices == 0) {
                    throw std::runtime_error("CUDA device requested but no devices found.");
                }
                for (auto device_string : devices) {
                    // Use most of GPU mem but leave some for buffer.
                    auto caller = create_cuda_caller(model_config, model_path, chunk_size,
                                                     batch_size, device_string, 0.9f);
                    for (size_t i = 0; i < num_runners; i++) {
                        runners.push_back(std::make_shared<CudaModelRunner>(caller));
                    }
                    if (runners.back()->batch_size() != batch_size) {
                        spdlog::debug("- set batch size for {} to {}", device_string,
                                      runners.back()->batch_size());
                    }
                }

                for (auto device_string : devices) {
                    // The fraction argument for GPU memory allocates the fraction of the
                    // _remaining_ memory to the caller. So, we allocate all of the available
                    // memory after simplex caller has been instantiated to the duplex caller.
                    // ALWAYS auto tune the duplex batch size (i.e. batch_size passed in is 0.)
                    auto caller = create_cuda_caller(stereo_model_config, stereo_model_path,
                                                     chunk_size, 0, device_string, 1.f);
                    for (size_t i = 0; i < num_runners; i++) {
                        stereo_runners.push_back(std::make_shared<CudaModelRunner>(caller));
                    }
                }
            }
#endif  // __APPLE__
#endif  // DORADO_GPU_BUILD
            spdlog::info("> Starting Stereo Duplex pipeline");

            auto stereo_model_stride = stereo_runners.front()->model_stride();

            auto adjusted_stereo_overlap = (overlap / stereo_model_stride) * stereo_model_stride;

            const int kStereoBatchTimeoutMS = 5000;
            auto stereo_basecaller_node = std::make_unique<BasecallerNode>(
                    read_filter_node, std::move(stereo_runners), adjusted_stereo_overlap,
                    kStereoBatchTimeoutMS, duplex_rg_name, 1000, "StereoBasecallerNode", true);
            auto simplex_model_stride = runners.front()->model_stride();

            StereoDuplexEncoderNode stereo_node =
                    StereoDuplexEncoderNode(*stereo_basecaller_node, simplex_model_stride);

            PairingNode pairing_node(stereo_node,
                                     template_complement_map.empty()
                                             ? std::optional<std::map<std::string, std::string>>{}
                                             : template_complement_map);

            // Initialize duplex split settings and create a duplex split node
            // with the given settings and number of devices. If
            // splitter_settings.enabled is set to false, the splitter node will
            // act as a passthrough, meaning it won't perform any splitting
            // operations and will just pass data through.
            DuplexSplitSettings splitter_settings;
            DuplexSplitNode splitter_node(pairing_node, splitter_settings, num_devices);

            auto adjusted_simplex_overlap = (overlap / simplex_model_stride) * simplex_model_stride;

            const int kSimplexBatchTimeoutMS = 100;
            auto basecaller_node = std::make_unique<BasecallerNode>(
                    splitter_node, std::move(runners), adjusted_simplex_overlap,
                    kSimplexBatchTimeoutMS, model, 1000, "BasecallerNode", true);

            ScalerNode scaler_node(*basecaller_node, model_config.signal_norm_params,
                                   num_devices * 2);

            DataLoader loader(scaler_node, "cpu", num_devices, 0, std::move(read_list));

            // Setup stats counting
            using dorado::stats::make_stats_reporter;
            stats_reporters.push_back(make_stats_reporter(*stereo_basecaller_node));
            stats_reporters.push_back(make_stats_reporter(stereo_node));
            stats_reporters.push_back(make_stats_reporter(pairing_node));
            stats_reporters.push_back(make_stats_reporter(splitter_node));
            stats_reporters.push_back(make_stats_reporter(*basecaller_node));
            stats_reporters.push_back(make_stats_reporter(loader));
            stats_reporters.push_back(make_stats_reporter(scaler_node));

            constexpr auto kStatsPeriod = 100ms;
            auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables);
            // End stats counting setup.

            loader.load_reads(reads, parser.get<bool>("--recursive"), DataLoader::BY_CHANNEL);
            bam_writer->join();  // Explicitly wait for all output rows to be written.
            stats_sampler->terminate();
        }
        tracker.summarize();
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        return 1;
    }
    return 0;
}
}  // namespace dorado
