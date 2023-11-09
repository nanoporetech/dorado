#include "Version.h"
#include "cli/cli_utils.h"
#include "data_loader/DataLoader.h"
#include "models/models.h"
#include "nn/CRFModelConfig.h"
#include "nn/Runners.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/DuplexReadTaggingNode.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/Pipelines.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"
#include "utils/duplex_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/sys_stats.h"
#include "utils/torch_utils.h"
#include "utils/types.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <thread>
#include <unordered_set>

namespace dorado {

using dorado::utils::default_parameters;
using namespace std::chrono_literals;

int duplex(int argc, char* argv[]) {
    using dorado::utils::default_parameters;
    utils::InitLogging();
    // TODO: Re-enable torch deterministic for duplex after OOM
    // on smaller VRAM GPUs is fixed.
    // The issue appears to be that enabling deterministic algorithms
    // through torch requires a larger CUBLAS workspace to be configured.
    // This larger CUBLAS workspace is causing memory fragmentation in
    // the CUDACaching allocator since the workspace memory is always cached
    // and there's no threadsafe way to clear the workspace memory for a
    // specific CUBLAS handle/stream combination through public APIs.
    // Details: When the first set of simplex basecalls
    // happen, the caching allocator is able to allocate memory for
    // inference + decode + CUBLAS workspace. Then when the stereo model
    // is run the first time, the caching allocator finds enough memory for
    // that model's inference + decode + CUBLAS workspace because the memory
    // footprint for the stero model is much smaller. However, when the next
    // simplex call is run on the same GPU, the allocator can't find enough
    // contiguous unreserved memory for the simplex
    // inference and decode step because of the fragmentation caused by the
    // cached CUBLAS workspace from the stero model. This causes OOM.
    //utils::make_torch_deterministic();
    torch::set_num_threads(1);

    cli::ArgParser parser("dorado");
    parser.visible.add_argument("model").help("Model");
    parser.visible.add_argument("reads").help(
            "Reads in POD5 format or BAM/SAM format for basespace.");
    parser.visible.add_argument("--pairs")
            .default_value(std::string(""))
            .help("Space-delimited csv containing read ID pairs. If not provided, pairing will be "
                  "performed automatically");
    parser.visible.add_argument("--emit-fastq").default_value(false).implicit_value(true);
    parser.visible.add_argument("--emit-sam")
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("-t", "--threads").default_value(0).scan<'i', int>();

    parser.visible.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\" etc..")
            .default_value(utils::default_parameters.device);

    parser.visible.add_argument("-b", "--batchsize")
            .default_value(default_parameters.batchsize)
            .scan<'i', int>()
            .help("if 0 an optimal batchsize will be selected. batchsizes are rounded to the "
                  "closest multiple of 64.");

    parser.visible.add_argument("-c", "--chunksize")
            .default_value(default_parameters.chunksize)
            .scan<'i', int>();

    parser.visible.add_argument("-o", "--overlap")
            .default_value(default_parameters.overlap)
            .scan<'i', int>();

    parser.visible.add_argument("-r", "--recursive")
            .default_value(false)
            .implicit_value(true)
            .help("Recursively scan through directories to load FAST5 and POD5 files");

    parser.visible.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to basecall. If not provided, all "
                  "reads will be basecalled")
            .default_value(std::string(""));

    parser.visible.add_argument("--min-qscore")
            .help("Discard reads with mean Q-score below this threshold.")
            .default_value(0)
            .scan<'i', int>();

    parser.visible.add_argument("--reference")
            .help("Path to reference for alignment.")
            .default_value(std::string(""));

    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    cli::add_minimap2_arguments(parser, AlignerNode::dflt_options);
    cli::add_internal_arguments(parser);

    try {
        cli::parse(parser, argc, argv);

        auto device(parser.visible.get<std::string>("-x"));
        auto model(parser.visible.get<std::string>("model"));

        if (model.find("fast") != std::string::npos) {
            spdlog::warn("Fast models are currently not recommended for duplex basecalling.");
        }

        auto reads(parser.visible.get<std::string>("reads"));
        std::string pairs_file = parser.visible.get<std::string>("--pairs");
        auto threads = static_cast<size_t>(parser.visible.get<int>("--threads"));
        auto min_qscore(parser.visible.get<int>("--min-qscore"));
        auto ref = parser.visible.get<std::string>("--reference");
        const bool basespace_duplex = (model.compare("basespace") == 0);
        std::vector<std::string> args(argv, argv + argc);
        if (parser.visible.get<bool>("--verbose")) {
            utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
        }
        std::map<std::string, std::string> template_complement_map;
        auto read_list = utils::load_read_list(parser.visible.get<std::string>("--read-ids"));

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

        bool emit_moves = false, duplex = true;

        auto output_mode = HtsWriter::OutputMode::BAM;

        auto emit_fastq = parser.visible.get<bool>("--emit-fastq");
        auto emit_sam = parser.visible.get<bool>("--emit-sam");

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

        bool recursive_file_loading = parser.visible.get<bool>("--recursive");

        const std::string dump_stats_file = parser.hidden.get<std::string>("--dump_stats_file");
        const std::string dump_stats_filter = parser.hidden.get<std::string>("--dump_stats_filter");
        const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);

        size_t num_reads = (basespace_duplex ? read_list_from_pairs.size()
                                             : DataLoader::get_num_reads(reads, read_list, {},
                                                                         recursive_file_loading));
        spdlog::debug("> Reads to process: {}", num_reads);

        SamHdrPtr hdr(sam_hdr_init());
        cli::add_pg_hdr(hdr.get(), args);

        PipelineDescriptor pipeline_desc;
        auto hts_writer = PipelineDescriptor::InvalidNodeHandle;
        auto aligner = PipelineDescriptor::InvalidNodeHandle;
        auto converted_reads_sink = PipelineDescriptor::InvalidNodeHandle;
        if (ref.empty()) {
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, "-", output_mode, 4, num_reads);
            converted_reads_sink = hts_writer;
        } else {
            auto options = cli::process_minimap2_arguments(parser, AlignerNode::dflt_options);
            aligner = pipeline_desc.add_node<AlignerNode>({}, ref, options,
                                                          std::thread::hardware_concurrency());
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, "-", output_mode, 4, num_reads);
            pipeline_desc.add_node_sink(aligner, hts_writer);
            converted_reads_sink = aligner;
        }
        auto read_converter = pipeline_desc.add_node<ReadToBamType>(
                {converted_reads_sink}, emit_moves, 2, 0.0f, nullptr, 1000);
        auto duplex_read_tagger = pipeline_desc.add_node<DuplexReadTaggingNode>({read_converter});
        // The minimum sequence length is set to 5 to avoid issues with duplex node printing very short sequences for mismatched pairs.
        std::unordered_set<std::string> read_ids_to_filter;
        auto read_filter_node = pipeline_desc.add_node<ReadFilterNode>(
                {duplex_read_tagger}, min_qscore, default_parameters.min_sequence_length,
                read_ids_to_filter, 5);

        std::vector<dorado::stats::StatsCallable> stats_callables;
        ProgressTracker tracker(int(num_reads), duplex);
        stats_callables.push_back(
                [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
        stats::NamedStats final_stats;
        std::unique_ptr<dorado::stats::StatsSampler> stats_sampler;
        std::vector<dorado::stats::StatsReporter> stats_reporters{dorado::stats::sys_stats_report};

        std::unique_ptr<dorado::Pipeline> pipeline;
        constexpr auto kStatsPeriod = 100ms;

        if (basespace_duplex) {  // Execute a Basespace duplex pipeline.
            if (pairs_file.empty()) {
                spdlog::error("The --pairs argument is required for the basespace model.");
                return 1;  // Exit with an error code
            }

            spdlog::info("> Loading reads");
            auto read_map = read_bam(reads, read_list_from_pairs);

            spdlog::info("> Starting Basespace Duplex Pipeline");
            threads = threads == 0 ? std::thread::hardware_concurrency() : threads;

            pipeline_desc.add_node<BaseSpaceDuplexCallerNode>({read_filter_node},
                                                              std::move(template_complement_map),
                                                              std::move(read_map), threads);

            pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
            if (pipeline == nullptr) {
                spdlog::error("Failed to create pipeline");
                std::exit(EXIT_FAILURE);
            }

            // Write header as no read group info is needed.
            auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
            hts_writer_ref.set_and_write_header(hdr.get());

            constexpr auto kStatsPeriod = 100ms;
            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);
        } else {  // Execute a Stereo Duplex pipeline.

            const auto model_path = std::filesystem::canonical(std::filesystem::path(model));
            model = model_path.filename().string();
            auto model_config = load_crf_model_config(model_path);

            if (!DataLoader::is_read_data_present(reads, recursive_file_loading)) {
                std::string err = "No POD5 or FAST5 data found in path: " + reads;
                throw std::runtime_error(err);
            }

            // Check sample rate of model vs data.
            auto data_sample_rate = DataLoader::get_sample_rate(reads, recursive_file_loading);
            auto model_sample_rate = model_config.sample_rate;
            if (model_sample_rate < 0) {
                // If unsuccessful, find sample rate by model name.
                model_sample_rate = models::get_sample_rate_by_model_name(
                        models::extract_model_from_model_path(model_path.string()));
            }
            auto skip_model_compatibility_check =
                    parser.hidden.get<bool>("--skip-model-compatibility-check");
            if (!skip_model_compatibility_check &&
                !sample_rates_compatible(data_sample_rate, model_sample_rate)) {
                std::stringstream err;
                err << "Sample rate for model (" << model_sample_rate << ") and data ("
                    << data_sample_rate << ") are not compatible.";
                throw std::runtime_error(err.str());
            }
            auto stereo_model_name = utils::get_stereo_model_name(model, data_sample_rate);
            const auto stereo_model_path =
                    model_path.parent_path() / std::filesystem::path(stereo_model_name);

            if (!std::filesystem::exists(stereo_model_path)) {
                if (!models::download_models(model_path.parent_path().u8string(),
                                             stereo_model_name)) {
                    throw std::runtime_error("Failed to download model: " + stereo_model_name);
                }
            }
            auto stereo_model_config = load_crf_model_config(stereo_model_path);

            // Write read group info to header.
            auto duplex_rg_name = std::string(model + "_" + stereo_model_name);
            auto read_groups = DataLoader::load_read_groups(reads, model, recursive_file_loading);
            read_groups.merge(
                    DataLoader::load_read_groups(reads, duplex_rg_name, recursive_file_loading));
            std::vector<std::string> barcode_kits;
            utils::add_rg_hdr(hdr.get(), read_groups, barcode_kits, nullptr);

            int batch_size(parser.visible.get<int>("-b"));
            int chunk_size(parser.visible.get<int>("-c"));
            int overlap(parser.visible.get<int>("-o"));
            const size_t num_runners = default_parameters.num_runners;

            int stereo_batch_size = 0;
#if DORADO_GPU_BUILD
#ifdef __APPLE__
            if (device == "metal") {
                // For now, the minimal batch size is used for the duplex model.
                stereo_batch_size = 48;
            }
#endif
#endif
            // Note: The memory assignment between simplex and duplex callers have been
            // performed based on empirical results considering a SUP model for simplex
            // calling.
            auto [runners, num_devices] = create_basecall_runners(
                    model_config, device, num_runners, 0, batch_size, chunk_size, 0.9f, true);

            std::vector<Runner> stereo_runners;
            // The fraction argument for GPU memory allocates the fraction of the
            // _remaining_ memory to the caller. So, we allocate all of the available
            // memory after simplex caller has been instantiated to the duplex caller.
            // ALWAYS auto tune the duplex batch size (i.e. batch_size passed in is 0.)
            // except for on metal
            // WORKAROUND: As a workaround to CUDA OOM, force stereo to have a smaller
            // memory footprint for both model and decode function. This will increase the
            // chances for the stereo model to use the cached allocations from the simplex
            // model.
            std::tie(stereo_runners, std::ignore) =
                    create_basecall_runners(stereo_model_config, device, num_runners, 0,
                                            stereo_batch_size, chunk_size, 0.5f, true);

            spdlog::info("> Starting Stereo Duplex pipeline");

            PairingParameters pairing_parameters;
            if (template_complement_map.empty()) {
                pairing_parameters =
                        DuplexPairingParameters{ReadOrder::BY_CHANNEL, DEFAULT_DUPLEX_CACHE_DEPTH};
            } else {
                pairing_parameters = std::move(template_complement_map);
            }

            auto mean_qscore_start_pos = model_config.mean_qscore_start_pos;
            if (mean_qscore_start_pos < 0) {
                mean_qscore_start_pos =
                        models::get_mean_qscore_start_pos_by_model_name(stereo_model_name);
                if (mean_qscore_start_pos < 0) {
                    throw std::runtime_error("Mean q-score start position cannot be < 0");
                }
            }
            pipelines::create_stereo_duplex_pipeline(
                    pipeline_desc, std::move(runners), std::move(stereo_runners), overlap,
                    mean_qscore_start_pos, int(num_devices * 2), int(num_devices),
                    std::move(pairing_parameters), read_filter_node);

            pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
            if (pipeline == nullptr) {
                spdlog::error("Failed to create pipeline");
                std::exit(EXIT_FAILURE);
            }

            // At present, header output file header writing relies on direct node method calls
            // rather than the pipeline framework.
            auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
            if (!ref.empty()) {
                const auto& aligner_ref =
                        dynamic_cast<AlignerNode&>(pipeline->get_node_ref(aligner));
                utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
            }
            hts_writer_ref.set_and_write_header(hdr.get());

            DataLoader loader(*pipeline, "cpu", num_devices, 0, std::move(read_list));

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

            // Run pipeline.
            loader.load_reads(reads, parser.visible.get<bool>("--recursive"),
                              ReadOrder::BY_CHANNEL);
        }

        // Wait for the pipeline to complete.  When it does, we collect
        // final stats to allow accurate summarisation.
        final_stats = pipeline->terminate(DefaultFlushOptions());

        // Stop the stats sampler thread before tearing down any pipeline objects.
        stats_sampler->terminate();

        tracker.update_progress_bar(final_stats);
        tracker.summarize();
        if (!dump_stats_file.empty()) {
            std::ofstream stats_file(dump_stats_file);
            stats_sampler->dump_stats(stats_file,
                                      dump_stats_filter.empty()
                                              ? std::nullopt
                                              : std::optional<std::regex>(dump_stats_filter));
        }
    } catch (const std::exception& e) {
        spdlog::error(e.what());
        return 1;
    }
    return 0;
}
}  // namespace dorado
