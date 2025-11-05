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
#include "demux/KitInfoProvider.h"
#include "demux/adapter_info.h"
#include "demux/barcoding_info.h"
#include "demux/parse_custom_kit.h"
#include "demux/parse_custom_sequences.h"
#include "file_info/file_info.h"
#include "hts_utils/bam_utils.h"
#include "hts_utils/hts_file.h"
#include "hts_writer/HtsFileWriter.h"
#include "hts_writer/HtsFileWriterBuilder.h"
#include "model_resolver/ModelResolver.h"
#include "model_resolver/Models.h"
#include "models/models.h"
#include "poly_tail/poly_tail_calculator_selector.h"
#include "read_pipeline/base/DefaultClientInfo.h"
#include "read_pipeline/nodes/AdapterDetectorNode.h"
#include "read_pipeline/nodes/AlignerNode.h"
#include "read_pipeline/nodes/BarcodeClassifierNode.h"
#include "read_pipeline/nodes/PolyACalculatorNode.h"
#include "read_pipeline/nodes/ReadFilterNode.h"
#include "read_pipeline/nodes/ReadToBamTypeNode.h"
#include "read_pipeline/nodes/TrimmerNode.h"
#include "read_pipeline/nodes/WriterNode.h"
#include "resume_loader/ResumeLoader.h"
#include "torch_utils/torch_utils.h"
#include "utils/SampleSheet.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/benchmark_timer.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/string_utils.h"
#include "utils/sys_stats.h"

#include <argparse/argparse.hpp>
#include <cxxpool.h>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/utils.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if DORADO_CUDA_BUILD
#include "torch_utils/cuda_utils.h"
#endif

using dorado::utils::default_parameters;
using OutputMode = dorado::utils::HtsFile::OutputMode;
using namespace std::chrono_literals;
using namespace dorado::models;
using namespace dorado::model_resolution;
using namespace dorado::config;
namespace fs = std::filesystem;

namespace dorado {

namespace {

/**
 * Class caching directory entries for a folder, along with the path.
 */
class InputPod5FolderInfo final {
    const std::filesystem::path m_data_path;
    const DataLoader::InputFiles m_pod5_files;

public:
    InputPod5FolderInfo(std::filesystem::path data_path, DataLoader::InputFiles pod5_files)
            : m_data_path(std::move(data_path)), m_pod5_files(std::move(pod5_files)) {}
    const std::filesystem::path& path() const { return m_data_path; }
    const DataLoader::InputFiles& files() const { return m_pod5_files; }
};

void set_dorado_basecaller_args(argparse::ArgumentParser& parser, int& verbosity) {
    parser.add_argument("model").help(
            "Model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory.");
    parser.add_argument("data").help("The data directory or POD5 file path.");

    // Default "Optional arguments" group
    parser.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    cli::add_device_arg(parser);

    parser.add_argument("--models-directory")
            .default_value(std::string("."))
            .help("Optional directory to search for existing models or download new models "
                  "into.");

    {
        parser.add_group("Input data arguments");
        parser.add_argument("-r", "--recursive")
                .flag()
                .help("Recursively scan through directories to load POD5 files.");
        parser.add_argument("-l", "--read-ids")
                .help("A file with a newline-delimited list of reads to basecall. If not provided, "
                      "all reads will be basecalled.")
                .default_value(std::string{});
        parser.add_argument("-n", "--max-reads")
                .help("Limit the number of reads to be basecalled.")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("--resume-from")
                .help("Resume basecalling from the given HTS file. Fully written read records are "
                      "not processed again.")
                .default_value(std::string{});
        parser.add_argument("--disable-read-splitting").flag().help("Disable read splitting");
    }
    {
        parser.add_group("Output arguments");
        parser.add_argument("--min-qscore")
                .help("Discard reads with mean Q-score below this threshold or write them to "
                      "output files marked `fail` if `--output-dir` is set.")
                .default_value(0)
                .scan<'i', int>();
        parser.add_argument("--emit-moves").help("Write the move table to the 'mv' tag.").flag();
        cli::add_basecaller_output_arguments(parser);
    }
    {
        parser.add_group("Alignment arguments");
        parser.add_argument("--reference")
                .help("Path to reference for alignment.")
                .default_value(std::string{});
        parser.add_argument("--bed-file")
                .help("Optional bed-file. If specified, overlaps between the alignments and "
                      "bed-file entries will be counted, and recorded in BAM output using the 'bh' "
                      "read tag.")
                .default_value(std::string{});
        alignment::mm2::add_options_string_arg(parser);
    }
    {
        const std::string mods_codes = utils::join(models::modified_model_variants(), ", ");
        parser.add_group("Modified model arguments");
        auto& modbase_mutex_group = parser.add_mutually_exclusive_group();
        modbase_mutex_group.add_argument("--modified-bases")
                .help("A space separated list of modified base codes. Choose from: " + mods_codes +
                      ".")
                .nargs(argparse::nargs_pattern::at_least_one)
                .action([mods_codes](const std::string& value) {
                    const auto& mods = models::modified_model_variants();
                    if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                        spdlog::error("'{}' is not a supported modification please select from {}",
                                      value, mods_codes);
                        std::exit(EXIT_FAILURE);
                    }
                    return value;
                });
        modbase_mutex_group.add_argument("--modified-bases-models")
                .default_value(std::string{})
                .help("A comma separated list of modified base model paths.");
        parser.add_argument("--modified-bases-threshold")
                .scan<'f', float>()
                .help("The minimum predicted methylation probability for a modified base to be "
                      "emitted in an all-context model, [0, 1].");
        parser.add_argument("--modified-bases-batchsize")
                .scan<'i', int>()
                .help("The modified base models batch size.");
    }
    {
        parser.add_group("Barcoding arguments");
        parser.add_argument("--kit-name")
                .help("Enable barcoding with the provided kit name. Choose from: " +
                      dorado::barcode_kits::barcode_kits_list_str() + ".")
                .default_value(std::string{});
        parser.add_argument("--sample-sheet")
                .help("Path to the sample sheet to use.")
                .default_value(std::string{});
        parser.add_argument("--barcode-both-ends")
                .help("Require both ends of a read to be barcoded for a double ended barcode.")
                .flag();
        parser.add_argument("--barcode-arrangement")
                .help("Path to file with custom barcode arrangement. Requires --kit-name.");
        parser.add_argument("--barcode-sequences")
                .help("Path to file with custom barcode sequences. Requires --kit-name and "
                      "--barcode-arrangement.");
        const std::string extended_primer_codes = utils::join(demux::extended_primer_names(), ", ");
        parser.add_argument("--primer-sequences")
                .help("Path to fasta file with custom primer sequences, or the name of a supported "
                      "3rd-party primer set. If specifying a supported primer set, choose from: " +
                      extended_primer_codes + ".");
    }
    {
        parser.add_group("Trimming arguments");
        parser.add_argument("--no-trim")
                .help("Skip trimming of barcodes, adapters, and primers. If option is not chosen, "
                      "trimming of all three is enabled.")
                .flag();
        parser.add_argument("--trim")
                .help("Specify what to trim. Options are 'none', 'all', and 'adapters'. The "
                      "default behaviour is to trim all detected adapters, primers, and barcodes. "
                      "Choose 'adapters' to just trim adapters. The 'none' choice is equivelent to "
                      "using --no-trim. Note that this only applies to DNA. RNA adapters are "
                      "always trimmed.")
                .default_value(std::string{});
        parser.add_argument("--rna-adapters").hidden().help("Force use of RNA adapters.").flag();
    }
    {
        parser.add_group("Poly(A) arguments");
        parser.add_argument("--estimate-poly-a")
                .help("Estimate poly(A)/poly(T) tail lengths (beta feature). Primarily meant for "
                      "cDNA and dRNA use cases.")
                .flag();
        parser.add_argument("--poly-a-config")
                .help("Configuration file for poly(A) estimation to change default behaviours")
                .default_value(std::string{});
    }
    {
        parser.add_group("Advanced arguments");
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
        parser.add_argument("--disable-variable-chunk-sizes").flag().hidden();
        parser.add_argument("--overlap")
                .hidden()
                .help("The number of samples overlapping neighbouring chunks.")
                .default_value(default_parameters.overlap)
                .scan<'i', int>();
    }
    cli::add_internal_arguments(parser);
}

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

void terminate_runners(std::vector<dorado::basecall::RunnerPtr>& runners,
                       std::vector<dorado::modbase::RunnerPtr>& modbase_runners) {
    for (auto& runner : runners) {
        runner->terminate();
    }
    for (auto& runner : modbase_runners) {
        runner->terminate();
    }
}

Models load_basecaller_models(const argparse::ArgumentParser& parser,
                              const InputPod5FolderInfo& pod5_folder_info,
                              const std::string& context) {
    try {
        BasecallerModelResolver resolver{
                parser.get<std::string>("model"),
                parser.get<std::string>("--modified-bases-models"),
                parser.get<std::vector<std::string>>("--modified-bases"),
                cli::get_optional_argument<std::string>("--models-directory", parser),
                parser.get<bool>("--skip-model-compatibility-check"),
                pod5_folder_info.files().get(),
        };

        return Models(resolver.resolve());
    } catch (const std::exception& e) {
        spdlog::error("Failed to resolve {} models: {}", context, e.what());
        std::exit(EXIT_FAILURE);
    }
}

void setup(const std::vector<std::string>& args,
           const Models& models,
           const InputPod5FolderInfo& pod5_folder_info,
           const std::string& device,
           const std::string& ref,
           const std::string& bed,
           size_t num_runners,
           const ModBaseBatchParams& modbase_params,
           const std::optional<std::string>& output_dir,
           bool emit_fastq,
           bool emit_sam,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           const std::string& read_list_file_path,
           const alignment::Minimap2Options& aligner_options,
           const std::string& dump_stats_file,
           const std::string& dump_stats_filter,
           bool run_batchsize_benchmarks,
           bool emit_batchsize_benchmarks,
           const std::string& resume_from_file,
           bool enable_read_splitting,
           [[maybe_unused]] bool variable_chunk_sizes,
           bool estimate_poly_a,
           const std::string& polya_config,
           const std::shared_ptr<const dorado::demux::BarcodingInfo>& barcoding_info,
           const std::shared_ptr<const dorado::demux::AdapterInfo>& adapter_info,
           const int run_for_arg) {
    const BasecallModelConfig& model_config = models.get_simplex_config();
    spdlog::trace(model_config.to_string());
    spdlog::trace(modbase_params.to_string());

    if (!file_info::is_pod5_data_present(pod5_folder_info.files().get())) {
        std::string err = "No POD5 data found in path: " + pod5_folder_info.path().string();
        throw std::runtime_error(err);
    }

    auto read_list = utils::load_read_list(read_list_file_path);
    size_t num_reads = file_info::get_num_reads(pod5_folder_info.files().get(), read_list, {});
    if (num_reads == 0) {
        spdlog::error("No reads found in path: " + pod5_folder_info.path().string());
        std::exit(EXIT_FAILURE);
    }
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    ProgressTracker tracker(ProgressTracker::SIMPLEX, num_reads);

    const bool enable_aligner = !ref.empty();

#if DORADO_CUDA_BUILD
    auto initial_device_info = utils::get_cuda_device_info(device, false);
    cli::log_requested_cuda_devices(initial_device_info);
#endif

    // create modbase runners first so basecall runners can pick batch sizes based on available memory
    auto modbase_runners = api::create_modbase_runners(models.get_modbase_model_paths(), device,
                                                       modbase_params.runners_per_caller,
                                                       modbase_params.batchsize);

    std::vector<basecall::RunnerPtr> runners;
    size_t num_devices = 0;
#if DORADO_CUDA_BUILD
    if (device != "cpu") {
        // Iterate over the separate devices to create the basecall runners.
        // We may have multiple GPUs with different amounts of free memory left after the modbase runners were created.
        // This allows us to set a different memory_limit_fraction in case we have a heterogeneous GPU setup
        auto updated_device_info = utils::get_cuda_device_info(device, false);
        std::vector<std::pair<std::string, float>> gpu_fractions;
        std::vector<int> device_ids;
        for (size_t i = 0; i < updated_device_info.size(); ++i) {
            auto device_id = "cuda:" + std::to_string(updated_device_info[i].device_id);
            auto fraction = static_cast<float>(updated_device_info[i].free_mem) /
                            static_cast<float>(initial_device_info[i].free_mem);
            gpu_fractions.push_back(std::make_pair(device_id, fraction));
            device_ids.push_back(updated_device_info[i].device_id);
        }

        if (variable_chunk_sizes &&
            !api::check_variable_chunk_sizes_supported(model_config, device_ids)) {
            variable_chunk_sizes = false;
        }

        cxxpool::thread_pool pool{gpu_fractions.size()};
        struct BasecallerRunners {
            std::vector<dorado::basecall::RunnerPtr> runners;
            size_t num_devices{};
        };

        std::vector<std::future<BasecallerRunners>> futures;
        auto create_runners = [&](const std::string& device_id, float fraction) {
            BasecallerRunners basecaller_runners;
            std::tie(basecaller_runners.runners, basecaller_runners.num_devices) =
                    api::create_basecall_runners(
                            {model_config, device_id, fraction, api::PipelineType::simplex, 0.f,
                             run_batchsize_benchmarks, emit_batchsize_benchmarks,
                             variable_chunk_sizes},
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
            num_devices += data.num_devices;
        }

        if (num_devices == 0) {
            throw std::runtime_error("CUDA device requested but no devices found.");
        }
    } else
#endif
    {
        std::tie(runners, num_devices) = api::create_basecall_runners(
                {model_config, device, 1.f, api::PipelineType::simplex, 0.f,
                 run_batchsize_benchmarks, emit_batchsize_benchmarks, false},
                num_runners, 0);
    }

    auto read_groups = file_info::load_read_groups(
            pod5_folder_info.files().get(), models.get_simplex_model_name(),
            utils::join(models.get_modbase_model_names(), ","));

    const bool adapter_trimming_enabled =
            (adapter_info && (adapter_info->trim_adapters || adapter_info->trim_primers));
    const auto thread_allocations = utils::default_thread_allocations(
            int(num_devices), !modbase_runners.empty() ? int(modbase_params.threads) : 0,
            enable_aligner, barcoding_info != nullptr, adapter_trimming_enabled);

    std::string gpu_names{};
#if DORADO_CUDA_BUILD
    gpu_names = utils::get_cuda_gpu_names(device);
#endif

    SamHdrPtr hdr(sam_hdr_init());
    cli::add_pg_hdr(hdr.get(), "basecaller", args, device);

    if (barcoding_info) {
        utils::add_rg_headers_with_barcode_kit(hdr.get(), read_groups, barcoding_info->kit_name,
                                               barcoding_info->sample_sheet.get());
    } else {
        utils::add_rg_headers(hdr.get(), read_groups);
    }
    std::vector<std::unique_ptr<hts_writer::IWriter>> writers;
    {
        auto progress_callback = utils::ProgressCallback([&tracker](size_t progress) {
            tracker.update_post_processing_progress(static_cast<float>(progress));
        });
        auto description_callback =
                utils::DescriptionCallback([&tracker](const std::string& description) {
                    tracker.set_description(description);
                });
        auto hts_writer_builder = hts_writer::BasecallHtsFileWriterBuilder(
                emit_fastq, emit_sam, !ref.empty(), output_dir, thread_allocations.writer_threads,
                progress_callback, description_callback, gpu_names);

        if (hts_writer_builder.get_output_mode() == OutputMode::FASTQ && !modbase_runners.empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with modbase models as FASTQ cannot store modbase "
                    "results.");
            terminate_runners(runners, modbase_runners);
            std::exit(EXIT_FAILURE);
        }

        std::unique_ptr<hts_writer::HtsFileWriter> hts_file_writer = hts_writer_builder.build();
        if (hts_file_writer == nullptr) {
            spdlog::error("Failed to create hts file writer");
            terminate_runners(runners, modbase_runners);
            std::exit(EXIT_FAILURE);
        }

        tracker.set_post_processing_percentage(hts_file_writer->finalise_is_noop() ? 0.0f : 0.5f);
        writers.push_back(std::move(hts_file_writer));
    }

    PipelineDescriptor pipeline_desc;
    auto hts_writer = pipeline_desc.add_node<WriterNode>({}, std::move(writers));
    auto aligner = PipelineDescriptor::InvalidNodeHandle;
    auto current_sink_node = hts_writer;
    if (enable_aligner) {
        auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
        auto bed_file_access = std::make_shared<alignment::BedFileAccess>();
        if (!bed.empty()) {
            if (!bed_file_access->load_bedfile(bed)) {
                throw std::runtime_error("Could not load bed-file " + bed);
            }
        }
        aligner = pipeline_desc.add_node<AlignerNode>({current_sink_node}, index_file_access,
                                                      bed_file_access, ref, bed, aligner_options,
                                                      thread_allocations.aligner_threads);
        current_sink_node = aligner;
    }
    current_sink_node = pipeline_desc.add_node<ReadToBamTypeNode>(
            {current_sink_node}, emit_moves, thread_allocations.read_converter_threads,
            modbase_params.threshold, 1000, min_qscore);

    {
        // When writing to output, write reads below min_qscore to "fail"
        const size_t maybe_min_qscore = output_dir.has_value() ? 0 : min_qscore;

        current_sink_node = pipeline_desc.add_node<ReadFilterNode>(
                {current_sink_node}, maybe_min_qscore, default_parameters.min_sequence_length,
                std::unordered_set<std::string>{}, thread_allocations.read_filter_threads);
    }

    if ((barcoding_info && barcoding_info->trim) || adapter_trimming_enabled) {
        current_sink_node = pipeline_desc.add_node<TrimmerNode>({current_sink_node}, 1);
    }

    const bool is_rna_adapter =
            is_rna_model(model_config) &&
            (adapter_info->rna_adapters || (barcoding_info && !barcoding_info->kit_name.empty()));

    auto client_info = std::make_shared<DefaultClientInfo>();
    client_info->contexts().register_context<const demux::AdapterInfo>(adapter_info);

    if (estimate_poly_a) {
        poly_tail::PolyTailCalibrationCoeffs calibration{
                .speed = model_config.polya_speed_correction,
                .offset = model_config.polya_offset_correction};
        auto poly_tail_calc_selector =
                std::make_shared<const poly_tail::PolyTailCalculatorSelector>(
                        polya_config, is_rna_model(model_config), is_rna_adapter, calibration);
        if (poly_tail_calc_selector->has_enabled_calculator()) {
            client_info->contexts().register_context<const poly_tail::PolyTailCalculatorSelector>(
                    poly_tail_calc_selector);
            current_sink_node = pipeline_desc.add_node<PolyACalculatorNode>(
                    {current_sink_node}, std::thread::hardware_concurrency(), 1000);
        }
    }
    if (barcoding_info) {
        client_info->contexts().register_context<const demux::BarcodingInfo>(barcoding_info);
        current_sink_node = pipeline_desc.add_node<BarcodeClassifierNode>(
                {current_sink_node}, thread_allocations.barcoder_threads);
    }
    if (adapter_trimming_enabled) {
        current_sink_node = pipeline_desc.add_node<AdapterDetectorNode>(
                {current_sink_node}, thread_allocations.adapter_threads);
    }

    auto mean_qscore_start_pos = model_config.mean_qscore_start_pos;

    api::create_simplex_pipeline(pipeline_desc, std::move(runners), std::move(modbase_runners),
                                 mean_qscore_start_pos, thread_allocations.scaler_node_threads,
                                 enable_read_splitting, thread_allocations.splitter_node_threads,
                                 thread_allocations.modbase_threads, current_sink_node,
                                 PipelineDescriptor::InvalidNodeHandle);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters{dorado::stats::sys_stats_report};
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework - because we must guarantee that the header is set
    // BEFORE we write any reads.
    if (enable_aligner) {
        const auto& aligner_ref = pipeline->get_node_ref<AlignerNode>(aligner);
        utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
    }

    {
        // Set the sam header for all writers
        const auto& hts_writer_ref = pipeline->get_node_ref<WriterNode>(hts_writer);
        hts_writer_ref.set_shared_header(std::move(hdr));
    }

    std::unordered_set<std::string> reads_already_processed;
    if (!resume_from_file.empty()) {
        if (output_dir.has_value()) {
            spdlog::error("--resume-from cannot be used with --output-dir.");
            std::exit(EXIT_FAILURE);
        }

        spdlog::info("> Inspecting resume file...");
        // Turn off warning logging as header info is fetched.
        auto initial_hts_log_level = hts_get_log_level();
        hts_set_log_level(HTS_LOG_OFF);
        auto pg_keys =
                utils::extract_pg_keys_from_hdr(resume_from_file, {"CL"}, "ID", "basecaller");
        hts_set_log_level(initial_hts_log_level);

        std::vector<std::string> tokens;
        try {
            tokens = cli::extract_token_from_cli(pg_keys["CL"]);
        } catch (const std::exception& e) {
            spdlog::debug("Caught error: '{}'", e.what());
            spdlog::error(
                    "Failed to parse resume parameters as --resume-from file 'CL' (Command Line) "
                    "header is invalid. This might happen if the HTS file headers were dropped "
                    "with the default samtools '--no-headers' argument.");
            std::exit(EXIT_FAILURE);
        }

        // First token is the dorado binary name. Remove that because the
        // sub parser only knows about the `basecaller` command.
        tokens.erase(tokens.begin());

        std::vector<std::string> resume_args_excluding_mm2_opts{};
        alignment::mm2::extract_options_string_arg(tokens, resume_args_excluding_mm2_opts);

        // Create a new basecaller parser to parse the resumed basecaller CLI string
        argparse::ArgumentParser resume_parser("dorado");
        int verbosity = 0;
        set_dorado_basecaller_args(resume_parser, verbosity);
        resume_parser.parse_known_args(resume_args_excluding_mm2_opts);

        const Models resume_models =
                load_basecaller_models(resume_parser, pod5_folder_info, "--resume-from");

        if (resume_models != models) {
            spdlog::error(
                    "Inconsistent models used in this pipeline and those used in the --resume-from "
                    "file.");
            models.print("Current");
            resume_models.print("Resumed");
            std::exit(EXIT_FAILURE);
        }

        // Resume functionality injects reads directly into the writer node.
        auto& hts_writer_ref = pipeline->get_node_ref<WriterNode>(hts_writer);
        ResumeLoader resume_loader(hts_writer_ref, resume_from_file);
        resume_loader.copy_completed_reads();
        reads_already_processed = resume_loader.get_processed_read_ids();
    }

    tracker.reset_initialization_time();
    tracker.set_description("Basecalling");

    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

    // If we are doing benchmarking, set the time-limit and register a handler to deal with
    // stats reporting.
    std::unique_ptr<BenchmarkTimer> benchmark_timer_ptr{};
    if (run_for_arg > 0) {
        ShutdownCallback shutdown_callback = [&pipeline]() {
            pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::Yes});
            spdlog::info("Benchmarking time-limit reached. Shutting down.");
        };
        benchmark_timer_ptr =
                std::make_unique<BenchmarkTimer>(run_for_arg * 1000, std::move(shutdown_callback));
    }

    DataLoader loader(*pipeline, "cpu", thread_allocations.loader_threads, max_reads, read_list,
                      reads_already_processed);

    auto func = [client_info](ReadCommon& read) { read.client_info = client_info; };
    loader.add_read_initialiser(func);

    // This is blocking on all reads
    loader.load_reads(pod5_folder_info.files(), ReadOrder::UNRESTRICTED);

    // Wait for the pipeline to complete.  When it does, we collect final stats to allow accurate summarisation.
    // Note that if the pipeline was already terminated by the ShutdownCallback, this does nothing, and final_stats
    // will be empty.
    auto final_stats = pipeline->terminate({.fast = utils::AsyncQueueTerminateFast::No});

    // Stop the stats sampler thread before tearing down any pipeline objects.
    // Then update progress tracking one more time from this thread, to
    // allow accurate summarisation.
    stats_sampler->terminate();
    if (!final_stats.empty()) {
        tracker.update_progress_bar(final_stats);
    }
    // Give the user a nice summary.
    tracker.summarize();
    if (!dump_stats_file.empty()) {
        std::ofstream stats_file(dump_stats_file);
        stats_sampler->dump_stats(stats_file,
                                  dump_stats_filter.empty()
                                          ? std::nullopt
                                          : std::optional<std::regex>(dump_stats_filter));
    }
}

}  // namespace

int basecaller(int argc, char* argv[]) {
    utils::initialise_torch();
    utils::set_torch_allocator_max_split_size();
    utils::make_torch_deterministic();

    argparse::ArgumentParser parser("dorado", DORADO_VERSION, argparse::default_arguments::help);
    int verbosity = 0;
    set_dorado_basecaller_args(parser, verbosity);

    std::vector<std::string> args_excluding_mm2_opts{};
    auto mm2_option_string = alignment::mm2::extract_options_string_arg({argv, argv + argc},
                                                                        args_excluding_mm2_opts);

    try {
        cli::parse(parser, args_excluding_mm2_opts);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    const std::filesystem::path data_path = parser.get<std::string>("data");
    const bool recursive_file_loading = parser.get<bool>("--recursive");

    DataLoader::InputFiles input_pod5s;
    try {
        input_pod5s = DataLoader::InputFiles::search_pod5s(data_path, recursive_file_loading);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load pod5 data: '{}'", e.what());
        return EXIT_FAILURE;
    }
    const InputPod5FolderInfo pod5_folder_info(data_path, std::move(input_pod5s));

    const auto device = cli::parse_device(parser);

    if (parser.get<std::string>("--reference").empty() &&
        !parser.get<std::string>("--bed-file").empty()) {
        spdlog::error("--bed-file cannot be used without --reference.");
        return EXIT_FAILURE;
    }

    Models models = load_basecaller_models(parser, pod5_folder_info, "basecaller");
    models.set_basecaller_batch_params(cli::get_batch_params(parser), device);

    bool trim_barcodes = true, trim_primers = true, trim_adapters = true;
    auto trim_options = parser.get<std::string>("--trim");
    if (parser.get<bool>("--no-trim")) {
        if (!trim_options.empty()) {
            spdlog::error("Only one of --no-trim and --trim can be used.");
            return EXIT_FAILURE;
        }
        trim_barcodes = trim_primers = trim_adapters = false;
    }
    if (trim_options == "none") {
        trim_barcodes = trim_primers = trim_adapters = false;
    } else if (trim_options == "adapters") {
        trim_barcodes = trim_primers = false;
    } else if (!trim_options.empty() && trim_options != "all") {
        spdlog::error("Unsupported --trim value '{}'.", trim_options);
        return EXIT_FAILURE;
    }

    std::string polya_config = "";
    if (parser.get<bool>("--estimate-poly-a")) {
        polya_config = parser.get<std::string>("--poly-a-config");
    }

    if (parser.is_used("--barcode-sequences")) {
        if (!parser.is_used("--kit-name") || !parser.is_used("--barcode-arrangement")) {
            spdlog::error("--barcode-sequences requires --barcode-arrangement and --kit-name.");
            return EXIT_FAILURE;
        }
    }

    if (parser.is_used("--barcode-arrangement")) {
        if (!parser.is_used("--kit-name")) {
            spdlog::error("--barcode-arrangement requires --kit-name.");
            return EXIT_FAILURE;
        }
    }

    std::shared_ptr<demux::BarcodingInfo> barcoding_info{};
    if (parser.is_used("--kit-name")) {
        barcoding_info = std::make_shared<demux::BarcodingInfo>();
        barcoding_info->kit_name = parser.get<std::string>("--kit-name");
        barcoding_info->barcode_both_ends = parser.get<bool>("--barcode-both-ends");
        barcoding_info->trim = trim_barcodes;

        if (!demux::try_configure_custom_barcode_sequences(
                    parser.present<std::string>("--barcode-sequences"))) {
            std::exit(EXIT_FAILURE);
        }

        if (!demux::try_configure_custom_barcode_arrangement(
                    parser.present<std::string>("--barcode-arrangement"))) {
            std::exit(EXIT_FAILURE);
        }

        auto barcode_sample_sheet = parser.get<std::string>("--sample-sheet");
        if (!barcode_sample_sheet.empty()) {
            barcoding_info->sample_sheet =
                    std::make_shared<const utils::SampleSheet>(barcode_sample_sheet, false);
            barcoding_info->allowed_barcodes = barcoding_info->sample_sheet->get_barcode_values();
        }

        if (!barcode_kits::is_valid_barcode_kit(barcoding_info->kit_name)) {
            spdlog::error(
                    "{} is not a valid barcode kit name. Please run the help "
                    "command to find out available barcode kits.",
                    barcoding_info->kit_name);
            std::exit(EXIT_FAILURE);
        }
    }

    auto adapter_info = std::make_shared<demux::AdapterInfo>();
    adapter_info->trim_adapters = trim_adapters;
    adapter_info->trim_primers = trim_primers;
    auto primer_sequences = parser.present<std::string>("--primer-sequences");
    if (primer_sequences) {
        if (!adapter_info->set_primer_sequences(*primer_sequences)) {
            std::exit(EXIT_FAILURE);
        }
    }
    adapter_info->rna_adapters = parser.get<bool>("--rna-adapters");
    if (barcoding_info && !barcoding_info->kit_name.empty()) {
        demux::KitInfoProvider provider(barcoding_info->kit_name);
        const barcode_kits::KitInfo& kit_info = provider.get_kit_info(barcoding_info->kit_name);
        adapter_info->rna_adapters = kit_info.rna_barcodes;
    }

    spdlog::info("> Creating basecall pipeline");

    std::string err_msg{};
    auto minimap_options = alignment::mm2::try_parse_options(mm2_option_string, err_msg);
    if (!minimap_options) {
        spdlog::error("{}\n{}", err_msg, alignment::mm2::get_help_message());
        return EXIT_FAILURE;
    }

    // Force on running of batchsize benchmarks if emission is on
    const bool run_batchsize_benchmarks = parser.get<bool>("--emit-batchsize-benchmarks") ||
                                          parser.get<bool>("--run-batchsize-benchmarks");

    size_t device_count = 1;
#if DORADO_CUDA_BUILD
    auto initial_device_info = utils::get_cuda_device_info(device, false);
    device_count = initial_device_info.size();
#endif

    const auto modbase_params =
            validate_modbase_params(models.get_modbase_model_paths(), parser, device_count);

    auto run_for_arg = parser.get<int>("--run-for");
    if (run_for_arg < 0) {
        spdlog::error("Invalid value for --run-for: {}", run_for_arg);
        return EXIT_FAILURE;
    }

    try {
        setup(args, models, pod5_folder_info, device, parser.get<std::string>("--reference"),
              parser.get<std::string>("--bed-file"), default_parameters.num_runners, modbase_params,
              cli::get_output_dir(parser), cli::get_emit_fastq(parser), cli::get_emit_sam(parser),
              parser.get<bool>("--emit-moves"), parser.get<int>("--max-reads"),
              parser.get<int>("--min-qscore"), parser.get<std::string>("--read-ids"),
              *minimap_options, parser.get<std::string>("--dump_stats_file"),
              parser.get<std::string>("--dump_stats_filter"), run_batchsize_benchmarks,
              parser.get<bool>("--emit-batchsize-benchmarks"),
              parser.get<std::string>("--resume-from"),
              !parser.get<bool>("--disable-read-splitting"),
              !parser.get<bool>("--disable-variable-chunk-sizes"),
              parser.get<bool>("--estimate-poly-a"), polya_config, std::move(barcoding_info),
              std::move(adapter_info), run_for_arg);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return EXIT_FAILURE;
    }

    spdlog::info("> Finished");
    return EXIT_SUCCESS;
}

}  // namespace dorado
