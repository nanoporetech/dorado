#include "alignment/minimap2_args.h"
#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "cli/basecall_output_args.h"
#include "cli/cli.h"
#include "cli/cli_utils.h"
#include "cli/model_resolution.h"
#include "config/BasecallModelConfig.h"
#include "config/ModBaseBatchParams.h"
#include "config/ModBaseModelConfig.h"
#include "data_loader/DataLoader.h"
#include "file_info/file_info.h"
#include "model_downloader/model_downloader.h"
#include "models/metadata.h"
#include "models/model_complex.h"
#include "models/models.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BaseSpaceDuplexCallerNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/DuplexReadTaggingNode.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "torch_utils/auto_detect_device.h"
#include "torch_utils/duplex_utils.h"
#include "torch_utils/torch_utils.h"
#include "utils/SampleSheet.h"
#include "utils/arg_parse_ext.h"
#include "utils/bam_utils.h"
#include "utils/basecaller_utils.h"
#include "utils/fs_utils.h"
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

BatchParams get_batch_params(argparse::ArgumentParser& arg) {
    BatchParams basecaller{};
    basecaller.update(BatchParams::Priority::CLI_ARG,
                      cli::get_optional_argument<int>("--chunksize", arg),
                      cli::get_optional_argument<int>("--overlap", arg),
                      cli::get_optional_argument<int>("--batchsize", arg));
    return basecaller;
}

struct DuplexModels {
    std::filesystem::path model_path;
    std::string model_name;
    BasecallModelConfig model_config;

    std::filesystem::path stereo_model;
    BasecallModelConfig stereo_model_config;
    std::string stereo_model_name;

    std::vector<std::filesystem::path> mods_model_paths;

    std::set<std::filesystem::path> temp_paths{};
};

// If given a model path, create a ModelComplexSearch by looking up the model info by name and extracting
// the chemistry, sampling rate etc. Ordinarily this would fail but the user should have provided a known
// simplex model otherwise there's no way to match a stereo model.
// Otherwise, the user passed a ModelComplex which is parsed and the data is inspected to find the conditions.
ModelComplexSearch get_model_search(const std::string& model_arg, const DirEntries& dir_entries) {
    const ModelComplex model_complex = model_resolution::parse_model_argument(model_arg);
    if (model_complex.is_path()) {
        const auto model_path = std::filesystem::weakly_canonical(model_complex.raw);

        if (!check_model_path(model_path)) {
            std::exit(EXIT_FAILURE);
        };

        if (is_modbase_model(model_path)) {
            spdlog::error(
                    "Specified model `{}` is not a simplex model but a modified bases model. Pass "
                    "modified bases model paths using `--modified-bases-models`",
                    model_path.string());
            std::exit(EXIT_FAILURE);
        }
        const auto model_name = model_path.filename().string();
        // Find the simplex model - it must a known model otherwise we cannot match a stereo model
        const auto model_info = get_simplex_model_info(model_name);
        // Pass the model's ModelVariant (e.g. HAC) in here so everything matches
        // There are no mods variants if model_arg is a path
        const auto inferred_selection =
                ModelComplex{to_string(model_info.simplex.variant), model_info.simplex, {}};

        // Return the searcher which hasn't needed to inspect any data
        return ModelComplexSearch(inferred_selection, model_info.chemistry, false);
    }

    // Inspect data to find chemistry.
    const auto chemistry = file_info::get_unique_sequencing_chemistry(dir_entries);
    return ModelComplexSearch(model_complex, chemistry, true);
}

DuplexModels load_models(const std::string& model_arg,
                         const std::vector<std::string>& mod_bases,
                         const std::string& mod_bases_models,
                         const std::string& stereo_model_arg,
                         const std::optional<std::filesystem::path>& model_directory,
                         const DirEntries& dir_entries,
                         const BatchParams& batch_params,
                         const bool skip_model_compatibility_check,
                         const std::string& device) {
    ModelComplexSearch model_search = get_model_search(model_arg, dir_entries);
    const ModelComplex inferred_model_complex = model_search.complex();

    if (!mods_model_arguments_valid(inferred_model_complex, mod_bases, mod_bases_models)) {
        std::exit(EXIT_FAILURE);
    }

    if (inferred_model_complex.model.variant == ModelVariant::FAST) {
        spdlog::warn("Duplex is not supported for fast models.");
    }

    std::filesystem::path stereo_model_path;
    if (!stereo_model_arg.empty()) {
        stereo_model_path = std::filesystem::path(stereo_model_arg);
        if (!std::filesystem::exists(stereo_model_path)) {
            spdlog::error("--stereo-model does not exist at: '{}'.",
                          stereo_model_path.string().c_str());
            std::exit(EXIT_FAILURE);
        }
    }

    std::filesystem::path model_path;
    std::vector<std::filesystem::path> mods_model_paths;

    model_downloader::ModelDownloader downloader(model_directory);

    // Cannot use inferred_selection as it has the ModelVariant set differently.
    if (ModelComplexParser::parse(model_arg).is_path()) {
        model_path = std::filesystem::canonical(std::filesystem::path(model_arg));

        if (stereo_model_path.empty()) {
            stereo_model_path = model_path.parent_path() / model_search.stereo().name;
            if (!std::filesystem::exists(stereo_model_path)) {
                stereo_model_path = downloader.get(model_search.stereo(), "stereo duplex");
            }
        }

        if (!skip_model_compatibility_check) {
            const auto model_config = load_model_config(model_path);
            const auto model_name = model_path.filename().string();
            const auto model_sample_rate = model_config.sample_rate < 0
                                                   ? get_sample_rate_by_model_name(model_name)
                                                   : model_config.sample_rate;
            bool inspect_ok = true;
            models::SamplingRate data_sample_rate = 0;
            try {
                data_sample_rate = file_info::get_sample_rate(dir_entries);
            } catch (const std::exception& e) {
                inspect_ok = false;
                spdlog::warn(
                        "Could not check that model sampling rate and data sampling rate match. "
                        "Proceed with caution. Reason: {}",
                        e.what());
            }
            if (inspect_ok) {
                check_sampling_rates_compatible(model_sample_rate, data_sample_rate);
            }
        }
        mods_model_paths =
                get_non_complex_mods_models(model_path, mod_bases, mod_bases_models, downloader);

    } else {
        try {
            model_path = downloader.get(model_search.simplex(), "simplex");
            if (stereo_model_path.empty()) {
                stereo_model_path = downloader.get(model_search.stereo(), "stereo duplex");
            }
            // Either get the mods from the model complex or resolve from --modified-bases args
            mods_model_paths = inferred_model_complex.has_mods_variant()
                                       ? downloader.get(model_search.mods(), "mods")
                                       : get_non_complex_mods_models(model_path, mod_bases,
                                                                     mod_bases_models, downloader);
        } catch (const std::exception&) {
            utils::clean_temporary_models(downloader.temporary_models());
            throw;
        }
    }

    const auto model_name = model_path.filename().string();
    auto model_config = load_model_config(model_path);
    model_config.basecaller.update(batch_params);
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
    auto stereo_model_config = load_model_config(stereo_model_path);
    stereo_model_config.basecaller.update(batch_params);
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
                        mods_model_paths,    downloader.temporary_models()};
}

ModBaseBatchParams validate_modbase_params(const std::vector<std::filesystem::path>& paths,
                                           utils::arg_parse::ArgParser& parser) {
    // Convert path to params.
    auto params = get_modbase_params(paths);

    // Allow user to override batchsize.
    if (auto modbase_batchsize = parser.visible.present<int>("--modified-bases-batchsize");
        modbase_batchsize.has_value()) {
        params.batchsize = *modbase_batchsize;
    }

    // Allow user to override threshold.
    if (auto methylation_threshold = parser.visible.present<float>("--modified-bases-threshold");
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

    utils::arg_parse::ArgParser parser("dorado");

    parser.visible.add_argument("model").help(
            "Model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory.");
    parser.visible.add_argument("reads").help(
            "Reads in POD5 format or BAM/SAM format for basespace.");

    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    cli::add_device_arg(parser);

    parser.visible.add_argument("--models-directory")
            .default_value(std::string("."))
            .help("Optional directory to search for existing models or download new models into.");
    {
        parser.visible.add_group("Input data arguments");
        parser.visible.add_argument("-r", "--recursive")
                .help("Recursively scan through directories to load FAST5 and POD5 files.")
                .default_value(false)
                .implicit_value(true);
        parser.visible.add_argument("-l", "--read-ids")
                .help("A file with a newline-delimited list of reads to basecall. If not provided, "
                      "all reads will be basecalled.")
                .default_value(std::string(""));
        parser.visible.add_argument("--pairs")
                .default_value(std::string(""))
                .help("Space-delimited csv containing read ID pairs. If not provided, pairing will "
                      "be performed automatically.");
    }
    {
        parser.visible.add_group("Output arguments");
        parser.visible.add_argument("--min-qscore")
                .help("Discard reads with mean Q-score below this threshold.")
                .default_value(0)
                .scan<'i', int>();
        cli::add_basecaller_output_arguments(parser);
    }
    {
        parser.visible.add_group("Alignment arguments");
        parser.visible.add_argument("--reference")
                .help("Path to reference for alignment.")
                .default_value(std::string(""));
        alignment::mm2::add_options_string_arg(parser);
        parser.visible.add_argument("--bed-file")
                .help("Optional bed-file. If specified, overlaps between the alignments and "
                      "bed-file "
                      "entries will be counted, and recorded in BAM output using the 'bh' read "
                      "tag.")
                .default_value(std::string(""));
    }
    {
        const std::string options = utils::join(models::modified_model_variants(), ", ");
        parser.visible.add_group("Modified model arguments");
        parser.visible.add_argument("--modified-bases")
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
        parser.visible.add_argument("--modified-bases-models")
                .help("A comma separated list of modified base models")
                .default_value(std::string());
        parser.visible.add_argument("--modified-bases-threshold")
                .help("The minimum predicted methylation probability for a modified base to be "
                      "emitted in an all-context model, [0, 1].")
                .scan<'f', float>();
        parser.visible.add_argument("--modified-bases-batchsize")
                .scan<'i', int>()
                .help("The modified base models batch size.");
    }
    {
        parser.visible.add_group("Advanced arguments");
        parser.visible.add_argument("-t", "--threads").default_value(0).scan<'i', int>();
        parser.visible.add_argument("-b", "--batchsize")
                .help("The number of chunks in a batch. If 0 an optimal batchsize will be "
                      "selected.")
                .default_value(default_parameters.batchsize)
                .scan<'i', int>();
        parser.visible.add_argument("-c", "--chunksize")
                .help("The number of samples in a chunk.")
                .default_value(default_parameters.chunksize)
                .scan<'i', int>();
        parser.visible.add_argument("--overlap")
                .help("The number of samples overlapping neighbouring chunks.")
                .default_value(default_parameters.overlap)
                .scan<'i', int>();
    }
    cli::add_internal_arguments(parser);

    parser.hidden.add_argument("--stereo-model")
            .help("Path to stereo model")
            .default_value(std::string(""));

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);

    std::set<fs::path> temp_model_paths;
    try {
        utils::arg_parse::parse(parser, args_excluding_mm2_opts);

        auto device{parser.visible.get<std::string>("-x")};
        if (!cli::validate_device_string(device)) {
            return EXIT_FAILURE;
        }
        if (device == cli::AUTO_DETECT_DEVICE) {
            device = utils::get_auto_detected_device();
        }

        auto model(parser.visible.get<std::string>("model"));

        auto reads(parser.visible.get<std::string>("reads"));
        std::string pairs_file = parser.visible.get<std::string>("--pairs");
        auto threads = static_cast<size_t>(parser.visible.get<int>("--threads"));
        auto min_qscore(parser.visible.get<int>("--min-qscore"));
        auto ref = parser.visible.get<std::string>("--reference");
        auto bed = parser.visible.get<std::string>("--bed-file");
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

        bool emit_moves = false;

        auto output_mode = OutputMode::BAM;

        if (parser.visible.get<std::string>("--reference").empty() &&
            !parser.visible.get<std::string>("--bed-file").empty()) {
            spdlog::error("--bed-file cannot be used without --reference.");
            return EXIT_FAILURE;
        }

        const std::string dump_stats_file = parser.hidden.get<std::string>("--dump_stats_file");
        const std::string dump_stats_filter = parser.hidden.get<std::string>("--dump_stats_filter");
        const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);

        const bool recursive_file_loading = parser.visible.get<bool>("--recursive");
        auto input_files = DataLoader::InputFiles::search(reads, recursive_file_loading);
        if (!input_files.has_value()) {
            // search() will have logged an error message for us.
            return EXIT_FAILURE;
        }

        size_t num_reads = 0;
        if (basespace_duplex) {
            num_reads = read_list_from_pairs.size();
        } else {
            num_reads = file_info::get_num_reads(input_files->get(), read_list, {});
            if (num_reads == 0) {
                spdlog::error("No POD5 or FAST5 reads found in path: " + reads);
                return EXIT_FAILURE;
            }
        }
        spdlog::debug("> Reads to process: {}", num_reads);

        SamHdrPtr hdr(sam_hdr_init());
        cli::add_pg_hdr(hdr.get(), "duplex", args, device);

        auto hts_file = cli::extract_hts_file(parser);
        if (!hts_file) {
            return EXIT_FAILURE;
        }
        constexpr int WRITER_THREADS = 4;
        hts_file->set_num_threads(WRITER_THREADS);

        PipelineDescriptor pipeline_desc;
        auto hts_writer = PipelineDescriptor::InvalidNodeHandle;
        auto aligner = PipelineDescriptor::InvalidNodeHandle;
        auto converted_reads_sink = PipelineDescriptor::InvalidNodeHandle;
        std::string gpu_names{};
#if DORADO_CUDA_BUILD
        gpu_names = utils::get_cuda_gpu_names(device);
#endif
        if (ref.empty()) {
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
            converted_reads_sink = hts_writer;
        } else {
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
            hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
            pipeline_desc.add_node_sink(aligner, hts_writer);
            converted_reads_sink = aligner;
        }

        const auto read_converter = pipeline_desc.add_node<ReadToBamTypeNode>(
                {converted_reads_sink}, emit_moves, 2, std::nullopt, nullptr, 1000);
        const auto duplex_read_tagger =
                pipeline_desc.add_node<DuplexReadTaggingNode>({read_converter});
        // The minimum sequence length is set to 5 to avoid issues with duplex node printing very short sequences for mismatched pairs.
        std::unordered_set<std::string> read_ids_to_filter;
        auto read_filter_node = pipeline_desc.add_node<ReadFilterNode>(
                {duplex_read_tagger}, min_qscore, default_parameters.min_sequence_length,
                read_ids_to_filter, 5);

        std::unique_ptr<dorado::Pipeline> pipeline;
        ProgressTracker tracker(ProgressTracker::Mode::DUPLEX, int(num_reads),
                                hts_file->finalise_is_noop() ? 0.f : 0.5f);
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
            hts_file->set_header(hdr.get());

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);
        } else {  // Execute a Stereo Duplex pipeline.
            if (!file_info::is_read_data_present(input_files->get())) {
                std::string err = "No POD5 or FAST5 data found in path: " + reads;
                throw std::runtime_error(err);
            }

            const std::string stereo_model_arg = parser.hidden.get<std::string>("--stereo-model");
            const auto batch_params = get_batch_params(parser.visible);
            const bool skip_model_compatibility_check =
                    parser.hidden.get<bool>("--skip-model-compatibility-check");

            const auto models_directory = model_resolution::get_models_directory(parser.visible);
            const DuplexModels models = load_models(
                    model, mod_bases, mod_bases_models, stereo_model_arg, models_directory,
                    input_files->get(), batch_params, skip_model_compatibility_check, device);

            temp_model_paths = models.temp_paths;

#if DORADO_CUDA_BUILD
            auto initial_device_info = utils::get_cuda_device_info(device, false);
#endif

            const auto modbase_params = validate_modbase_params(models.mods_model_paths, parser);

            // create modbase runners first so basecall runners can pick batch sizes based on available memory
            auto mod_base_runners = api::create_modbase_runners(models.mods_model_paths, device,
                                                                modbase_params.runners_per_caller,
                                                                modbase_params.batchsize);

            if (!mod_base_runners.empty() && output_mode == OutputMode::FASTQ) {
                throw std::runtime_error("Modified base models cannot be used with FASTQ output");
            }

            // Write read group info to header.
            auto duplex_rg_name = std::string(models.model_name + "_" + models.stereo_model_name);
            // TODO: supply modbase model names once duplex modbase is complete
            auto read_groups =
                    file_info::load_read_groups(input_files->get(), models.model_name, "");
            read_groups.merge(file_info::load_read_groups(input_files->get(), duplex_rg_name, ""));
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
                                    {models.model_config, device_id, 0.9f * fraction,
                                     api::PipelineType::duplex, 0.f, false, false},
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
                                    {models.stereo_model_config, device_id, 0.5f * fraction,
                                     api::PipelineType::duplex, 0.f, false, false},
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
                std::tie(runners, num_devices) =
                        api::create_basecall_runners({models.model_config, device, 0.9f,
                                                      api::PipelineType::duplex, 0.f, false, false},
                                                     num_runners, 0);
                std::tie(stereo_runners, std::ignore) =
                        api::create_basecall_runners({models.stereo_model_config, device, 0.5f,
                                                      api::PipelineType::duplex, 0.f, false, false},
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

            auto mean_qscore_start_pos = models.model_config.mean_qscore_start_pos;

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
            hts_file->set_header(hdr.get());

            DataLoader loader(*pipeline, "cpu", num_devices, 0, std::move(read_list), {});
            loader.add_read_initialiser(client_info_init_func);

            stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
                    kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

            // Run pipeline.
            loader.load_reads(*input_files, ReadOrder::BY_CHANNEL);

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
        hts_file->finalise([&](size_t progress) {
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
