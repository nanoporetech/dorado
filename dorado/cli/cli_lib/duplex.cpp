#include "ProgressTracker.h"
#include "alignment/minimap2_args.h"
#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "basecall_output_args.h"
#include "cli/cli.h"
#include "cli/utils/cli_utils.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseBatchParams.h"
#include "config/ModBaseModelConfig.h"
#include "data_loader/DataLoader.h"
#include "file_info/file_info.h"
#include "hts_utils/bam_utils.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "model_resolver/ModelResolver.h"
#include "model_resolver/Models.h"
#include "models/models.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/nodes/AlignerNode.h"
#include "read_pipeline/nodes/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/nodes/DuplexReadTaggingNode.h"
#include "read_pipeline/nodes/ReadFilterNode.h"
#include "read_pipeline/nodes/ReadToBamTypeNode.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "torch_utils/duplex_utils.h"
#include "torch_utils/torch_utils.h"
#include "utils/SampleSheet.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/string_utils.h"
#include "utils/sys_stats.h"
#include "utils/types.h"

#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/utils.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif

using OutputMode = dorado::utils::HtsFile::OutputMode;
namespace fs = std::filesystem;

namespace dorado {

namespace {

using namespace dorado::models;
using namespace dorado::model_resolution;
using namespace dorado::config;

using DirEntries = std::vector<std::filesystem::directory_entry>;

ModBaseBatchParams validate_modbase_params(const std::vector<std::filesystem::path>& paths,
                                           argparse::ArgumentParser& parser,
                                           size_t device_count) {
    // Convert path to params.
    auto params = get_modbase_params(paths, device_count);

    // Allow user to override batchsize.
    if (auto modbase_batchsize = parser.present<int>("--modified-bases-batchsize");
        modbase_batchsize.has_value()) {
        params.batchsize = *modbase_batchsize;
    }

    // Allow user to override threshold.
    if (auto methylation_threshold = parser.present<float>("--modified-bases-threshold");
        methylation_threshold.has_value()) {
        if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
            throw std::runtime_error("--modified-bases-threshold must be between 0 and 1.");
        }
        params.threshold = *methylation_threshold;
    }

    // Check that the paths are all valid.
    for (const auto& mb_path : paths) {
        if (!is_modbase_model(mb_path)) {
            throw std::runtime_error("Modified bases model not found in the model path at " +
                                     std::filesystem::weakly_canonical(mb_path).string());
        }
    }

    // All looks good.
    return params;
}

DuplexModels load_duplex_models(const argparse::ArgumentParser& parser,
                                const DataLoader::InputFiles& input_pod5_files,
                                const std::string& context) {
    try {
        DuplexModelResolver resolver{
                parser.get<std::string>("model"),
                parser.get<std::string>("--modified-bases-models"),
                parser.get<std::vector<std::string>>("--modified-bases"),
                cli::get_optional_argument<std::string>("--stereo-model", parser),
                cli::get_optional_argument<std::string>("--models-directory", parser),
                parser.get<bool>("--skip-model-compatibility-check"),
                input_pod5_files.get(),
        };

        return DuplexModels(resolver.resolve());
    } catch (const std::exception& e) {
        spdlog::error("Failed to resolve {} models: {}", context, e.what());
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

using namespace std::chrono_literals;

int duplex(int argc, char* argv[]) {
    using dorado::utils::default_parameters;
    utils::initialise_torch();
    utils::set_torch_allocator_max_split_size();
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

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);

    parser.add_argument("model").help(
            "Model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory.");
    parser.add_argument("reads").help("Reads in POD5 format or BAM/SAM format for basespace.");

    int verbosity = 0;
    parser.add_argument("-v", "--verbose")
            .flag()
            .action([&](const auto&) { ++verbosity; })
            .append();

    cli::add_device_arg(parser);

    parser.add_argument("--models-directory")
            .default_value(std::string("."))
            .help("Optional directory to search for existing models or download new models into.");
    {
        parser.add_group("Input data arguments");
        parser.add_argument("-r", "--recursive")
                .help("Recursively scan through directories to load POD5 files.")
                .flag();
        parser.add_argument("-l", "--read-ids")
                .help("A file with a newline-delimited list of reads to basecall. If not provided, "
                      "all reads will be basecalled.")
                .default_value(std::string(""));
        parser.add_argument("--pairs")
                .default_value(std::string(""))
                .help("Space-delimited csv containing read ID pairs. If not provided, pairing will "
                      "be performed automatically.");
    }
    {
        parser.add_group("Output arguments");
        parser.add_argument("--min-qscore")
                .help("Discard reads with mean Q-score below this threshold or write them to "
                      "output files marked `fail` if `--output-dir` is set.")
                .default_value(0)
                .scan<'i', int>();
        cli::add_basecaller_output_arguments(parser);
    }
    {
        parser.add_group("Alignment arguments");
        parser.add_argument("--reference")
                .help("Path to reference for alignment.")
                .default_value(std::string(""));
        alignment::mm2::add_options_string_arg(parser);
        parser.add_argument("--bed-file")
                .help("Optional bed-file. If specified, overlaps between the alignments and "
                      "bed-file "
                      "entries will be counted, and recorded in BAM output using the 'bh' read "
                      "tag.")
                .default_value(std::string(""));
    }
    {
        const std::string options = utils::join(models::modified_model_variants(), ", ");
        parser.add_group("Modified model arguments");
        auto& modbase_mutex_group = parser.add_mutually_exclusive_group();
        modbase_mutex_group.add_argument("--modified-bases")
                .help("A space separated list of modified base codes. Choose from: " + options +
                      ".")
                .nargs(argparse::nargs_pattern::at_least_one)
                .action([&options](const std::string& value) {
                    const auto& mods = models::modified_model_variants();
                    if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                        spdlog::error("'{}' is not a supported modification please select from {}.",
                                      value, options);
                        std::exit(EXIT_FAILURE);
                    }
                    return value;
                });
        modbase_mutex_group.add_argument("--modified-bases-models")
                .help("A comma separated list of modified base model names or paths.")
                .default_value(std::string());
        parser.add_argument("--modified-bases-threshold")
                .help("The minimum predicted methylation probability for a modified base to be "
                      "emitted in an all-context model, [0, 1].")
                .scan<'f', float>();
        parser.add_argument("--modified-bases-batchsize")
                .scan<'i', int>()
                .help("The modified base models batch size.");
    }
    {
        parser.add_group("Advanced arguments");
        parser.add_argument("-t", "--threads").default_value(0).scan<'i', int>();
        parser.add_argument("-b", "--batchsize")
                .help("The number of chunks in a batch. If 0 an optimal batchsize will be "
                      "selected.")
                .default_value(default_parameters.batchsize)
                .scan<'i', int>();
        parser.add_argument("-c", "--chunksize")
                .hidden()
                .help("The number of samples in a chunk.")
                .default_value(default_parameters.chunksize)
                .scan<'i', int>();
        parser.add_argument("--overlap")
                .hidden()
                .help("The number of samples overlapping neighbouring chunks.")
                .default_value(default_parameters.overlap)
                .scan<'i', int>();
    }
    cli::add_internal_arguments(parser);

    parser.add_argument("--stereo-model")
            .hidden()
            .help("Path to stereo model")
            .default_value(std::string(""));

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);

    try {
        cli::parse(parser, args_excluding_mm2_opts);

        const auto device = cli::parse_device(parser);

        auto reads(parser.get<std::string>("reads"));
        std::string pairs_file = parser.get<std::string>("--pairs");
        auto threads = static_cast<size_t>(parser.get<int>("--threads"));
        auto min_qscore(parser.get<int>("--min-qscore"));
        auto ref = parser.get<std::string>("--reference");
        auto bed = parser.get<std::string>("--bed-file");
        std::vector<std::string> args(argv, argv + argc);
        if (parser.get<bool>("--verbose")) {
            utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
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

        bool emit_moves = false;

        if (parser.get<std::string>("--reference").empty() &&
            !parser.get<std::string>("--bed-file").empty()) {
            spdlog::error("--bed-file cannot be used without --reference.");
            return EXIT_FAILURE;
        }

        const std::string dump_stats_file = parser.get<std::string>("--dump_stats_file");
        const std::string dump_stats_filter = parser.get<std::string>("--dump_stats_filter");
        const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);

        const bool recursive_file_loading = parser.get<bool>("--recursive");

        DataLoader::InputFiles input_pod5_files;
        try {
            input_pod5_files = DataLoader::InputFiles::search_pod5s(reads, recursive_file_loading);
        } catch (const std::exception& e) {
            spdlog::error("Failed to load pod5 data: '{}'", e.what());
            return EXIT_FAILURE;
        }

        const bool basespace_duplex = parser.get<std::string>("model") == "basespace";

        size_t num_reads = 0;
        if (basespace_duplex) {
            num_reads = read_list_from_pairs.size();
        } else {
            num_reads = file_info::get_num_reads(input_pod5_files.get(), read_list, {});
            if (num_reads == 0) {
                spdlog::error("No reads found in path: " + reads);
                return EXIT_FAILURE;
            }
        }
        spdlog::debug("> Reads to process: {}", num_reads);

        SamHdrPtr hdr(sam_hdr_init());
        cli::add_pg_hdr(hdr.get(), "duplex", args, device);

        std::string gpu_names{};
#if DORADO_CUDA_BUILD
        gpu_names = utils::get_cuda_gpu_names(device);
#endif

        ProgressTracker tracker(ProgressTracker::Mode::DUPLEX, num_reads);

        std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
        OutputMode writer_output_mode;
        {
            auto progress_callback = utils::ProgressCallback([&tracker](size_t progress) {
                tracker.update_post_processing_progress(static_cast<float>(progress));
            });
            auto description_callback =
                    utils::DescriptionCallback([&tracker](const std::string& description) {
                        tracker.set_description(description);
                    });

            constexpr int WRITER_THREADS = 4;
            auto hts_writer_builder = hts_writer::BasecallHtsFileWriterBuilder(
                    cli::get_emit_fastq(parser), cli::get_emit_sam(parser), !ref.empty(),
                    cli::get_output_dir(parser), WRITER_THREADS, progress_callback,
                    description_callback, gpu_names);

            std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
            if (hts_file_writer == nullptr) {
                spdlog::error("Failed to create hts file writer");
                std::exit(EXIT_FAILURE);
            }
            writer_output_mode = hts_file_writer->get_mode();
            tracker.set_post_processing_percentage(hts_file_writer->finalise_is_noop() ? 0.0f
                                                                                       : 0.5f);
            writers.push_back(std::move(hts_file_writer));
        }

        PipelineDescriptor pipeline_desc;
        auto hts_writer = PipelineDescriptor::InvalidNodeHandle;
        auto aligner = PipelineDescriptor::InvalidNodeHandle;
        auto converted_reads_sink = PipelineDescriptor::InvalidNodeHandle;

        hts_writer = pipeline_desc.add_node<WriterNode>({}, std::move(writers));
        converted_reads_sink = hts_writer;

        if (!ref.empty()) {
            std::string err_msg{};
            auto minimap_options = alignment::mm2::try_parse_options(mm2_option_string, err_msg);
            if (!minimap_options) {
                spdlog::error("{}\n{}", err_msg, alignment::mm2::get_help_message());
                return EXIT_FAILURE;
            }
            auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
            auto bed_file_access = std::make_shared<alignment::BedFileAccess>();
            if (!bed.empty()) {
                if (!bed_file_access->load_bedfile(bed)) {
                    throw std::runtime_error("Could not load bed-file " + bed);
                }
            }

            aligner = pipeline_desc.add_node<AlignerNode>({}, index_file_access, bed_file_access,
                                                          ref, bed, *minimap_options,
                                                          std::thread::hardware_concurrency());
            pipeline_desc.add_node_sink(aligner, converted_reads_sink);
            converted_reads_sink = aligner;
        }

        const auto read_converter = pipeline_desc.add_node<ReadToBamTypeNode>(
                {converted_reads_sink}, emit_moves, 2, std::nullopt, 1000, min_qscore);
        const auto duplex_read_tagger =
                pipeline_desc.add_node<DuplexReadTaggingNode>({read_converter});
        // The minimum sequence length is set to 5 to avoid issues with duplex node printing very short sequences for mismatched pairs.
        std::unordered_set<std::string> read_ids_to_filter;

        // When writing to output, write reads below min_qscore to "fail"
        const size_t maybe_min_qscore = cli::get_output_dir(parser).has_value() ? 0 : min_qscore;

        auto read_filter_node = pipeline_desc.add_node<ReadFilterNode>(
                {duplex_read_tagger}, maybe_min_qscore, default_parameters.min_sequence_length,
                read_ids_to_filter, 5);

        std::unique_ptr<dorado::Pipeline> pipeline;

        tracker.set_description("Running duplex");
        std::vector<dorado::stats::StatsCallable> stats_callables;
        stats_callables.push_back(
                [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
        stats::NamedStats final_stats;
        std::unique_ptr<dorado::stats::StatsSampler> stats_sampler;
        std::vector<dorado::stats::StatsReporter> stats_reporters{dorado::stats::sys_stats_report};

        constexpr auto kStatsPeriod = 100ms;

        auto default_client_info = std::make_shared<DefaultClientInfo>();
        auto client_info_init_func = [default_client_info](ReadCommon& read) {
            read.client_info = default_client_info;
        };

        if (basespace_duplex) {  // Execute a Basespace duplex pipeline.
            if (pairs_file.empty()) {
                spdlog::error("The --pairs argument is required for the basespace model.");
                return EXIT_FAILURE;  // Exit with an error code
            }

            auto mod_bases_models = parser.get<std::string>("--modified-bases-models");
            auto mod_bases = parser.get<std::vector<std::string>>("--modified-bases");
            if (!mod_bases.empty() || !mod_bases_models.empty()) {
                spdlog::error("Basespace duplex does not support modbase models");
                return EXIT_FAILURE;
            }

            spdlog::info("> Loading reads");
            auto read_map = read_bam(reads, read_list_from_pairs);

            for (auto& [key, read] : read_map) {
                client_info_init_func(read->read_common);
            }

            spdlog::info("> Starting Basespace Duplex Pipeline");
            threads = threads == 0 ? std::thread::hardware_concurrency() : threads;

            pipeline_desc.add_node<BaseSpaceDuplexCallerNode>({read_filter_node},
                                                              std::move(template_complement_map),
                                                              std::move(read_map), threads);

            pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
            if (pipeline == nullptr) {
                spdlog::error("Failed to create pipeline");
                return EXIT_FAILURE;
            }

            // Write header as no read group info is needed.
            const auto& hts_writer_ref = pipeline->get_node_ref<WriterNode>(hts_writer);
            hts_writer_ref.set_shared_header(std::move(hdr));

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);
        } else {  // Execute a Stereo Duplex pipeline.

            DuplexModels models = load_duplex_models(parser, input_pod5_files, "duplex");
            models.set_basecaller_batch_params(cli::get_batch_params(parser), device);

            size_t device_count = 1;
#if DORADO_CUDA_BUILD
            auto initial_device_info = utils::get_cuda_device_info(device, false);
            cli::log_requested_cuda_devices(initial_device_info);
            device_count = initial_device_info.size();
#endif

            const auto modbase_params =
                    validate_modbase_params(models.get_modbase_model_paths(), parser, device_count);

            // create modbase runners first so basecall runners can pick batch sizes based on available memory
            auto mod_base_runners = api::create_modbase_runners(
                    models.get_modbase_model_paths(), device, modbase_params.runners_per_caller,
                    modbase_params.batchsize);

            if (!mod_base_runners.empty() && writer_output_mode == OutputMode::FASTQ) {
                throw std::runtime_error("Modified base models cannot be used with FASTQ output");
            }

            // Write read group info to header.
            auto duplex_rg_name = std::string(models.get_simplex_model_name() + "_" +
                                              models.get_stereo_model_name());
            // TODO: supply modbase model names once duplex modbase is complete
            auto read_groups = file_info::load_read_groups(input_pod5_files.get(),
                                                           models.get_simplex_config().stride,
                                                           models.get_simplex_model_name(), "");
            read_groups.merge(file_info::load_read_groups(
                    input_pod5_files.get(), models.get_stereo_config().stride, duplex_rg_name, ""));

            utils::add_rg_headers(hdr.get(), read_groups);

            const size_t num_runners = default_parameters.num_runners;

            std::vector<basecall::RunnerPtr> runners;
            std::vector<basecall::RunnerPtr> stereo_runners;
            size_t num_devices = 0;
#if DORADO_CUDA_BUILD
            if (device != "cpu") {
                // Iterate over the separate devices to create the basecall runners.
                // We may have multiple GPUs with different amounts of free memory left after the modbase runners were created.
                // This allows us to set a different memory_limit_fraction in case we have a heterogeneous GPU setup
                auto updated_device_info = utils::get_cuda_device_info(device, false);
                std::vector<std::pair<std::string, float>> gpu_fractions;
                for (size_t i = 0; i < updated_device_info.size(); ++i) {
                    auto device_id = "cuda:" + std::to_string(updated_device_info[i].device_id);
                    auto fraction = static_cast<float>(updated_device_info[i].free_mem) /
                                    static_cast<float>(initial_device_info[i].free_mem);
                    gpu_fractions.push_back(std::make_pair(device_id, fraction));
                }

                cxxpool::thread_pool pool{gpu_fractions.size()};
                struct DuplexBasecallerRunners {
                    std::vector<dorado::basecall::RunnerPtr> runners;
                    std::vector<dorado::basecall::RunnerPtr> stereo_runners;
                    size_t num_devices{};
                };

                std::vector<std::future<DuplexBasecallerRunners>> futures;
                auto create_runners = [&](const std::string& device_id, float fraction) {
                    // Note: The memory assignment between simplex and duplex callers have been
                    // performed based on empirical results considering a SUP model for simplex
                    // calling.
                    DuplexBasecallerRunners basecaller_runners;
                    std::tie(basecaller_runners.runners, basecaller_runners.num_devices) =
                            api::create_basecall_runners(
                                    {models.get_simplex_config(), device_id, 0.9f * fraction,
                                     api::PipelineType::duplex, 0.f, false, false, false},
                                    num_runners, 0);

                    // The fraction argument for GPU memory allocates the fraction of the
                    // _remaining_ memory to the caller. So, we allocate all of the available
                    // memory after simplex caller has been instantiated to the duplex caller.
                    // ALWAYS auto tune the duplex batch size (i.e. batch_size passed in is 0.)
                    // except for on metal
                    // WORKAROUND: As a workaround to CUDA OOM, force stereo to have a smaller
                    // memory footprint for both model and decode function. This will increase the
                    // chances for the stereo model to use the cached allocations from the simplex
                    // model.
                    std::tie(basecaller_runners.stereo_runners, std::ignore) =
                            api::create_basecall_runners(
                                    {models.get_stereo_config(), device_id, 0.5f * fraction,
                                     api::PipelineType::duplex, 0.f, false, false, false},
                                    num_runners, 0);

                    return basecaller_runners;
                };

                futures.reserve(gpu_fractions.size());
                for (const auto& [device_id, fraction] : gpu_fractions) {
                    futures.push_back(pool.push(create_runners, std::cref(device_id), fraction));
                }

                for (auto& future : futures) {
                    auto data = future.get();
                    runners.insert(runners.end(), std::make_move_iterator(data.runners.begin()),
                                   std::make_move_iterator(data.runners.end()));
                    stereo_runners.insert(stereo_runners.end(),
                                          std::make_move_iterator(data.stereo_runners.begin()),
                                          std::make_move_iterator(data.stereo_runners.end()));
                    num_devices += data.num_devices;
                }

                if (num_devices == 0) {
                    throw std::runtime_error("CUDA device requested but no devices found.");
                }
            } else
#endif
            {
                std::tie(runners, num_devices) = api::create_basecall_runners(
                        {models.get_simplex_config(), device, 0.9f, api::PipelineType::duplex, 0.f,
                         false, false, false},
                        num_runners, 0);
                std::tie(stereo_runners, std::ignore) = api::create_basecall_runners(
                        {models.get_stereo_config(), device, 0.5f, api::PipelineType::duplex, 0.f,
                         false, false, false},
                        num_runners, 0);
            }

            spdlog::info("> Starting Stereo Duplex pipeline");

            PairingParameters pairing_parameters;
            if (template_complement_map.empty()) {
                pairing_parameters =
                        DuplexPairingParameters{ReadOrder::BY_CHANNEL, DEFAULT_DUPLEX_CACHE_DEPTH};
            } else {
                pairing_parameters = std::move(template_complement_map);
            }

            auto mean_qscore_start_pos = models.get_simplex_config().mean_qscore_start_pos;

            api::create_stereo_duplex_pipeline(
                    pipeline_desc, std::move(runners), std::move(stereo_runners),
                    std::move(mod_base_runners), mean_qscore_start_pos, int(num_devices * 2),
                    int(num_devices), int(modbase_params.threads * num_devices),
                    std::move(pairing_parameters), read_filter_node,
                    PipelineDescriptor::InvalidNodeHandle);

            pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
            if (pipeline == nullptr) {
                spdlog::error("Failed to create pipeline");
                return EXIT_FAILURE;
            }

            // Set modbase threshold now that we have the params.
            pipeline->get_node_ref<ReadToBamTypeNode>(read_converter)
                    .set_modbase_threshold(modbase_params.threshold);

            // At present, header output file header writing relies on direct node method calls
            // rather than the pipeline framework.
            if (!ref.empty()) {
                const auto& aligner_ref = pipeline->get_node_ref<AlignerNode>(aligner);
                utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
            }

            // Write header as no read group info is needed.
            const auto& hts_writer_ref = pipeline->get_node_ref<WriterNode>(hts_writer);
            hts_writer_ref.set_shared_header(std::move(hdr));

            DataLoader loader(*pipeline, "cpu", num_devices, 0, std::move(read_list), {});
            loader.add_read_initialiser(client_info_init_func);

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

            // Run pipeline.
            tracker.reset_initialization_time();
            loader.load_reads(input_pod5_files, ReadOrder::BY_CHANNEL);
        }

        // Wait for the pipeline to complete.  When it does, we collect
        // final stats to allow accurate summarisation.
        final_stats = pipeline->terminate({.fast = dorado::utils::AsyncQueueTerminateFast::No});

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
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
}  // namespace dorado
