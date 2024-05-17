#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "basecall/CRFModelConfig.h"
#include "cli/cli_utils.h"
#include "data_loader/DataLoader.h"
#include "data_loader/ModelFinder.h"
#include "dorado_version.h"
#include "models/kits.h"
#include "models/metadata.h"
#include "models/models.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/DuplexReadTaggingNode.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"

#include <optional>
#if DORADO_CUDA_BUILD
#include "utils/cuda_utils.h"
#endif
#include "utils/duplex_utils.h"
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/string_utils.h"
#include "utils/sys_stats.h"
#include "utils/torch_utils.h"
#include "utils/tty_utils.h"
#include "utils/types.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/utils.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

using OutputMode = dorado::utils::HtsFile::OutputMode;
namespace fs = std::filesystem;

namespace dorado {

namespace {

basecall::BasecallerParams get_basecaller_params(argparse::ArgumentParser& arg) {
    // Argparser has no method returning optional<T> if the argument is unset but it has a default value
    auto get_opt = [&arg](const std::string& name) {
        return arg.is_used(name) ? std::optional<int>(arg.get<int>(name)) : std::nullopt;
    };

    basecall::BasecallerParams basecaller{};
    basecaller.update(basecall::BasecallerParams::Priority::CLI_ARG, get_opt("--chunksize"),
                      get_opt("--overlap"), get_opt("--batchsize"));
    return basecaller;
}

struct DuplexModels {
    std::filesystem::path model_path;
    std::string model_name;
    basecall::CRFModelConfig model_config;

    std::filesystem::path stereo_model;
    basecall::CRFModelConfig stereo_model_config;
    std::string stereo_model_name;

    std::vector<std::filesystem::path> mods_model_paths;

    std::set<std::filesystem::path> temp_paths{};
};

// If given a model path, create a ModelFinder by looking up the model info by name and extracting
// the chemistry, sampling rate etc that way. Otherwise, the user passed a model complex which
// is parsed and the data is inspected to find the conditions.
ModelFinder get_model_finder(const std::string& model_arg,
                             const std::string& reads,
                             const bool recursive_file_loading) {
    const ModelSelection model_selection = cli::parse_model_argument(model_arg);
    if (model_selection.is_path()) {
        // Get the model name
        const auto model_path = std::filesystem::canonical(std::filesystem::path(model_arg));
        const auto model_name = model_path.filename().string();

        // Try to find the model
        const auto model_info = ModelFinder::get_simplex_model_info(model_name);

        // Pass the model's ModelVariant (e.g. HAC) in here so everything matches
        // There are no mods variants if model_arg is a path
        const auto inferred_selection = ModelSelection{
                models::to_string(model_info.simplex.variant), model_info.simplex, {}};

        // Return the ModelFinder which hasn't needed to inspect any data
        return ModelFinder{model_info.chemistry, inferred_selection, false};
    }

    // Model complex given, inspect data to find chemistry.
    return cli::model_finder(model_selection, reads, recursive_file_loading, true);
}

DuplexModels load_models(const std::string& model_arg,
                         const std::vector<std::string>& mod_bases,
                         const std::string& mod_bases_models,
                         const std::string& reads,
                         const basecall::BasecallerParams& basecaller_params,
                         const bool recursive_file_loading,
                         const bool skip_model_compatibility_check,
                         const std::string& device) {
    using namespace dorado::models;

    ModelFinder model_finder = get_model_finder(model_arg, reads, recursive_file_loading);
    const ModelSelection inferred_selection = model_finder.get_selection();

    auto ways = {inferred_selection.has_mods_variant(), !mod_bases.empty(),
                 !mod_bases_models.empty()};
    if (std::count(ways.begin(), ways.end(), true) > 1) {
        throw std::runtime_error(
                "Only one of --modified-bases, --modified-bases-models, or modified models set "
                "via models argument can be used at once");
    };

    if (inferred_selection.model.variant == ModelVariant::FAST) {
        spdlog::warn("Duplex is not supported for fast models.");
    }

    std::filesystem::path model_path;
    std::filesystem::path stereo_model_path;
    std::vector<std::filesystem::path> mods_model_paths;

    // Cannot use inferred_selection as it has the ModelVariant set differently.
    if (ModelComplexParser::parse(model_arg).is_path()) {
        model_path = std::filesystem::canonical(std::filesystem::path(model_arg));
        stereo_model_path = model_path.parent_path() / model_finder.get_stereo_model_name();
        if (!std::filesystem::exists(stereo_model_path)) {
            stereo_model_path = model_finder.fetch_stereo_model();
        }

        if (!skip_model_compatibility_check) {
            const auto model_config = basecall::load_crf_model_config(model_path);
            const auto model_name = model_path.filename().string();
            check_sampling_rates_compatible(model_name, reads, model_config.sample_rate,
                                            recursive_file_loading);
        }
        mods_model_paths =
                dorado::get_non_complex_mods_models(model_path, mod_bases, mod_bases_models);

    } else {
        try {
            model_path = model_finder.fetch_simplex_model();
            stereo_model_path = model_finder.fetch_stereo_model();
            // Either get the mods from the model complex or resolve from --modified-bases args
            mods_model_paths = inferred_selection.has_mods_variant()
                                       ? model_finder.fetch_mods_models()
                                       : dorado::get_non_complex_mods_models(model_path, mod_bases,
                                                                             mod_bases_models);
        } catch (const std::exception&) {
            utils::clean_temporary_models(model_finder.downloaded_models());
            throw;
        }
    }

    const auto model_name = model_finder.get_simplex_model_name();
    auto model_config = basecall::load_crf_model_config(model_path);
    model_config.basecaller.update(basecaller_params);
    model_config.normalise_basecaller_params();

    if (device == "cpu" && model_config.basecaller.batch_size() == 0) {
        // Force the batch size to 128
        model_config.basecaller.set_batch_size(128);
    }
#if DORADO_METAL_BUILD
    else if (device == "metal" && model_config.is_tx_model() &&
             model_config.basecaller.batch_size() == 0) {
        // TODO: Remove with implementation of autobatch size calcuiaton for macos tx
        model_config.basecaller.set_batch_size(32);
    }
#endif

    const auto stereo_model_name = stereo_model_path.filename().string();
    auto stereo_model_config = basecall::load_crf_model_config(stereo_model_path);
    stereo_model_config.basecaller.update(basecaller_params);
    stereo_model_config.normalise_basecaller_params();

#if DORADO_METAL_BUILD
    if (device == "metal" && stereo_model_config.is_lstm_model()) {
        // ALWAYS auto tune the duplex batch size (i.e. batch_size passed in is 0.)
        // EXCEPT for on metal
        // For now, the minimal batch size is used for the duplex model.
        stereo_model_config.basecaller.set_batch_size(48);
    } else if (device == "metal" && model_config.is_tx_model() &&
               stereo_model_config.basecaller.batch_size() == 0) {
        // TODO: Remove with implementation of autobatch size calcuiaton for macos tx
        stereo_model_config.basecaller.set_batch_size(32);
    }
#endif
    if (device == "cpu" && stereo_model_config.basecaller.batch_size() == 0) {
        stereo_model_config.basecaller.set_batch_size(128);
    }

    return DuplexModels{model_path,          model_name,
                        model_config,        stereo_model_path,
                        stereo_model_config, stereo_model_name,
                        mods_model_paths,    model_finder.downloaded_models()};
}

}  // namespace

using dorado::utils::default_parameters;
using namespace std::chrono_literals;

int duplex(int argc, char* argv[]) {
    using dorado::utils::default_parameters;
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
    torch::set_num_threads(1);

    cli::ArgParser parser("dorado");
    parser.visible.add_argument("model").help(
            "model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory");
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

    parser.visible.add_argument("--modified-bases")
            .nargs(argparse::nargs_pattern::at_least_one)
            .action([](const std::string& value) {
                const auto& mods = models::modified_model_variants();
                if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                    spdlog::error("'{}' is not a supported modification please select from {}",
                                  value, utils::join(mods, ", "));
                    std::exit(EXIT_FAILURE);
                }
                return value;
            });

    parser.visible.add_argument("--modified-bases-models")
            .default_value(std::string())
            .help("a comma separated list of modified base models");

    parser.visible.add_argument("--modified-bases-threshold")
            .default_value(default_parameters.methylation_threshold)
            .scan<'f', float>()
            .help("the minimum predicted methylation probability for a modified base to be emitted "
                  "in an all-context model, [0, 1]");

    cli::add_minimap2_arguments(parser, alignment::DEFAULT_MM_PRESET);
    cli::add_internal_arguments(parser);

    std::set<fs::path> temp_model_paths;
    try {
        cli::parse(parser, argc, argv);

        auto device(parser.visible.get<std::string>("-x"));
        auto model(parser.visible.get<std::string>("model"));

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

        auto mod_bases = parser.visible.get<std::vector<std::string>>("--modified-bases");
        auto mod_bases_models = parser.visible.get<std::string>("--modified-bases-models");

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

        auto output_mode = OutputMode::BAM;

        auto emit_fastq = parser.visible.get<bool>("--emit-fastq");
        auto emit_sam = parser.visible.get<bool>("--emit-sam");

        if (emit_fastq && emit_sam) {
            throw std::runtime_error("Only one of --emit-{fastq, sam} can be set (or none).");
        }

        if (emit_fastq) {
            if (!parser.visible.get<std::string>("--reference").empty()) {
                spdlog::error(
                        "--emit-fastq cannot be used with --reference as FASTQ cannot store "
                        "alignment results.");
                return EXIT_FAILURE;
            }
            spdlog::info(
                    " - Note: FASTQ output is not recommended as not all data can be preserved.");
            output_mode = OutputMode::FASTQ;
        } else if (emit_sam || utils::is_fd_tty(stdout)) {
            output_mode = OutputMode::SAM;
        } else if (utils::is_fd_pipe(stdout)) {
            output_mode = OutputMode::UBAM;
        }

        const std::string dump_stats_file = parser.hidden.get<std::string>("--dump_stats_file");
        const std::string dump_stats_filter = parser.hidden.get<std::string>("--dump_stats_filter");
        const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);

        bool recursive_file_loading = parser.visible.get<bool>("--recursive");

        size_t num_reads = 0;
        if (basespace_duplex) {
            num_reads = read_list_from_pairs.size();
        } else {
            num_reads = DataLoader::get_num_reads(reads, read_list, {}, recursive_file_loading);
            if (num_reads == 0) {
                spdlog::error("No POD5 or FAST5 reads found in path: " + reads);
                return EXIT_FAILURE;
            }
        }
        spdlog::debug("> Reads to process: {}", num_reads);

        SamHdrPtr hdr(sam_hdr_init());
        cli::add_pg_hdr(hdr.get(), args, device);

        constexpr int WRITER_THREADS = 4;
        utils::HtsFile hts_file("-", output_mode, WRITER_THREADS, false);

        PipelineDescriptor pipeline_desc;
        auto hts_writer = PipelineDescriptor::InvalidNodeHandle;
        auto aligner = PipelineDescriptor::InvalidNodeHandle;
        auto converted_reads_sink = PipelineDescriptor::InvalidNodeHandle;
        std::string gpu_names{};
#if DORADO_CUDA_BUILD
        gpu_names = utils::get_cuda_gpu_names(device);
#endif
        if (ref.empty()) {
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, gpu_names);
            converted_reads_sink = hts_writer;
        } else {
            auto options = cli::process_minimap2_arguments<alignment::Minimap2Options>(parser);
            auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
            aligner = pipeline_desc.add_node<AlignerNode>({}, index_file_access, ref, "", options,
                                                          std::thread::hardware_concurrency());
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, gpu_names);
            pipeline_desc.add_node_sink(aligner, hts_writer);
            converted_reads_sink = aligner;
        }
        auto read_converter = pipeline_desc.add_node<ReadToBamTypeNode>(
                {converted_reads_sink}, emit_moves, 2, 0.0f, nullptr, 1000);
        auto duplex_read_tagger = pipeline_desc.add_node<DuplexReadTaggingNode>({read_converter});
        // The minimum sequence length is set to 5 to avoid issues with duplex node printing very short sequences for mismatched pairs.
        std::unordered_set<std::string> read_ids_to_filter;
        auto read_filter_node = pipeline_desc.add_node<ReadFilterNode>(
                {duplex_read_tagger}, min_qscore, default_parameters.min_sequence_length,
                read_ids_to_filter, 5);

        std::unique_ptr<dorado::Pipeline> pipeline;
        ProgressTracker tracker(int(num_reads), duplex, hts_file.finalise_is_noop() ? 0.f : 0.5f);
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
            hts_file.set_header(hdr.get());

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);
        } else {  // Execute a Stereo Duplex pipeline.

            if (!DataLoader::is_read_data_present(reads, recursive_file_loading)) {
                std::string err = "No POD5 or FAST5 data found in path: " + reads;
                throw std::runtime_error(err);
            }

            const auto basecaller_params = get_basecaller_params(parser.visible);
            const bool skip_model_compatibility_check =
                    parser.hidden.get<bool>("--skip-model-compatibility-check");

            const DuplexModels models =
                    load_models(model, mod_bases, mod_bases_models, reads, basecaller_params,
                                recursive_file_loading, skip_model_compatibility_check, device);

            temp_model_paths = models.temp_paths;

            // create modbase runners first so basecall runners can pick batch sizes based on available memory
            auto mod_base_runners = api::create_modbase_runners(
                    models.mods_model_paths, device, default_parameters.mod_base_runners_per_caller,
                    default_parameters.remora_batchsize);

            if (!mod_base_runners.empty() && output_mode == OutputMode::FASTQ) {
                throw std::runtime_error("Modified base models cannot be used with FASTQ output");
            }

            // Write read group info to header.
            auto duplex_rg_name = std::string(models.model_name + "_" + models.stereo_model_name);
            // TODO: supply modbase model names once duplex modbase is complete
            auto read_groups = DataLoader::load_read_groups(reads, models.model_name, "",
                                                            recursive_file_loading);
            read_groups.merge(DataLoader::load_read_groups(reads, duplex_rg_name, "",
                                                           recursive_file_loading));
            utils::add_rg_headers(hdr.get(), read_groups);

            const size_t num_runners = default_parameters.num_runners;

            // Note: The memory assignment between simplex and duplex callers have been
            // performed based on empirical results considering a SUP model for simplex
            // calling.
            auto [runners, num_devices] =
                    api::create_basecall_runners(models.model_config, device, num_runners, 0, 0.9f,
                                                 api::PipelineType::duplex, 0.f);

            std::vector<basecall::RunnerPtr> stereo_runners;
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
                    api::create_basecall_runners(models.stereo_model_config, device, num_runners, 0,
                                                 0.5f, api::PipelineType::duplex, 0.f);

            spdlog::info("> Starting Stereo Duplex pipeline");

            PairingParameters pairing_parameters;
            if (template_complement_map.empty()) {
                pairing_parameters =
                        DuplexPairingParameters{ReadOrder::BY_CHANNEL, DEFAULT_DUPLEX_CACHE_DEPTH};
            } else {
                pairing_parameters = std::move(template_complement_map);
            }

            auto mean_qscore_start_pos = models.model_config.mean_qscore_start_pos;

            api::create_stereo_duplex_pipeline(
                    pipeline_desc, std::move(runners), std::move(stereo_runners),
                    std::move(mod_base_runners), mean_qscore_start_pos, int(num_devices * 2),
                    int(num_devices), int(default_parameters.remora_threads * num_devices),
                    std::move(pairing_parameters), read_filter_node,
                    PipelineDescriptor::InvalidNodeHandle);

            pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
            if (pipeline == nullptr) {
                spdlog::error("Failed to create pipeline");
                return EXIT_FAILURE;
            }

            // At present, header output file header writing relies on direct node method calls
            // rather than the pipeline framework.
            if (!ref.empty()) {
                const auto& aligner_ref =
                        dynamic_cast<AlignerNode&>(pipeline->get_node_ref(aligner));
                utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
            }
            hts_file.set_header(hdr.get());

            DataLoader loader(*pipeline, "cpu", num_devices, 0, std::move(read_list), {});
            loader.add_read_initialiser(client_info_init_func);

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

            // Run pipeline.
            loader.load_reads(reads, parser.visible.get<bool>("--recursive"),
                              ReadOrder::BY_CHANNEL);

            utils::clean_temporary_models(temp_model_paths);
        }

        // Wait for the pipeline to complete.  When it does, we collect
        // final stats to allow accurate summarisation.
        final_stats = pipeline->terminate(DefaultFlushOptions());

        // Stop the stats sampler thread before tearing down any pipeline objects.
        stats_sampler->terminate();
        tracker.update_progress_bar(final_stats);

        // Report progress during output file finalisation.
        tracker.set_description("Sorting output files");
        hts_file.finalise([&](size_t progress) {
            tracker.update_post_processing_progress(static_cast<float>(progress));
        });

        tracker.summarize();
        if (!dump_stats_file.empty()) {
            std::ofstream stats_file(dump_stats_file);
            stats_sampler->dump_stats(stats_file,
                                      dump_stats_filter.empty()
                                              ? std::nullopt
                                              : std::optional<std::regex>(dump_stats_filter));
        }
    } catch (const std::exception& e) {
        utils::clean_temporary_models(temp_model_paths);
        spdlog::error(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
}  // namespace dorado
