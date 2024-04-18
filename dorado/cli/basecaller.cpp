#include "api/pipeline_creation.h"
#include "api/runner_creation.h"
#include "basecall/CRFModelConfig.h"
#include "cli/cli_utils.h"
#include "data_loader/DataLoader.h"
#include "data_loader/ModelFinder.h"
#include "demux/parse_custom_sequences.h"
#include "dorado_version.h"
#include "models/kits.h"
#include "models/models.h"
#include "read_pipeline/AdapterDetectorNode.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/DefaultClientInfo.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/PolyACalculatorNode.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ResumeLoaderNode.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#if DORADO_CUDA_BUILD
#include "utils/cuda_utils.h"
#endif
#include "utils/fs_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/parse_custom_kit.h"
#include "utils/stats.h"
#include "utils/string_utils.h"
#include "utils/sys_stats.h"
#include "utils/torch_utils.h"
#include "utils/tty_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>
#include <torch/utils.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>

namespace dorado {

namespace {

const barcode_kits::KitInfo& get_barcode_kit_info(const std::string& kit_name) {
    const auto kit_info = barcode_kits::get_kit_info(kit_name);
    if (!kit_info) {
        spdlog::error(
                "{} is not a valid barcode kit name. Please run the help "
                "command to find out available barcode kits.",
                kit_name);
        std::exit(EXIT_FAILURE);
    }
    return *kit_info;
}

std::pair<std::string, barcode_kits::KitInfo> get_custom_barcode_kit_info(
        const std::string& custom_kit_file) {
    auto custom_kit_info = barcode_kits::parse_custom_arrangement(custom_kit_file);
    if (!custom_kit_info) {
        spdlog::error("Unable to load custom barcode arrangement file: {}", custom_kit_file);
        std::exit(EXIT_FAILURE);
    }
    return *custom_kit_info;
}

}  // namespace

using dorado::utils::default_parameters;
using OutputMode = dorado::utils::HtsFile::OutputMode;
using namespace std::chrono_literals;
using namespace dorado::models;
namespace fs = std::filesystem;

void setup(const std::vector<std::string>& args,
           const fs::path& model_path,
           const std::string& data_path,
           const std::vector<fs::path>& remora_models,
           const std::string& device,
           const std::string& ref,
           size_t chunk_size,
           size_t overlap,
           size_t batch_size,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           float methylation_threshold_pct,
           OutputMode output_mode,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           const std::string& read_list_file_path,
           bool recursive_file_loading,
           const alignment::Minimap2Options& aligner_options,
           bool skip_model_compatibility_check,
           const std::string& dump_stats_file,
           const std::string& dump_stats_filter,
           const std::string& resume_from_file,
           const std::string& barcode_kit,
           bool barcode_both_ends,
           bool barcode_no_trim,
           bool adapter_no_trim,
           bool primer_no_trim,
           const std::string& barcode_sample_sheet,
           const std::optional<std::string>& custom_kit,
           const std::optional<std::string>& custom_barcode_file,
           const std::optional<std::string>& custom_primer_file,
           argparse::ArgumentParser& resume_parser,
           bool estimate_poly_a,
           const std::string& polya_config,
           const ModelSelection& model_selection) {
    const auto model_config = basecall::load_crf_model_config(model_path);

    spdlog::debug(model_config.to_string());
    const std::string model_name = models::extract_model_name_from_path(model_path);
    const std::string modbase_model_names = models::extract_model_names_from_paths(remora_models);

    if (!DataLoader::is_read_data_present(data_path, recursive_file_loading)) {
        std::string err = "No POD5 or FAST5 data found in path: " + data_path;
        throw std::runtime_error(err);
    }

    auto read_list = utils::load_read_list(read_list_file_path);
    size_t num_reads = DataLoader::get_num_reads(
            data_path, read_list, {} /*reads_already_processed*/, recursive_file_loading);
    if (num_reads == 0) {
        spdlog::error("No POD5 or FAST5 reads found in path: " + data_path);
        std::exit(EXIT_FAILURE);
    }
    num_reads = max_reads == 0 ? num_reads : std::min(num_reads, max_reads);

    // Sampling rate is checked by ModelFinder when a complex is given, only test for a path
    if (model_selection.is_path() && !skip_model_compatibility_check) {
        check_sampling_rates_compatible(model_name, data_path, model_config.sample_rate,
                                        recursive_file_loading);
    }

    if (is_rna_model(model_config)) {
        spdlog::info(
                " - BAM format does not support `U`, so RNA output files will include `T` instead "
                "of `U` for all file types.");
    }

    const bool enable_aligner = !ref.empty();

    // create modbase runners first so basecall runners can pick batch sizes based on available memory
    auto remora_runners = api::create_modbase_runners(
            remora_models, device, default_parameters.mod_base_runners_per_caller,
            remora_batch_size);

    auto [runners, num_devices] =
            api::create_basecall_runners(model_config, device, num_runners, 0, batch_size,
                                         chunk_size, 1.f, api::PipelineType::simplex, 0.f);

    auto read_groups = DataLoader::load_read_groups(data_path, model_name, modbase_model_names,
                                                    recursive_file_loading);

    const bool adapter_trimming_enabled = (!adapter_no_trim || !primer_no_trim);
    const bool barcode_enabled = !barcode_kit.empty() || custom_kit;
    const auto thread_allocations = utils::default_thread_allocations(
            int(num_devices), !remora_runners.empty() ? int(num_remora_threads) : 0, enable_aligner,
            barcode_enabled, adapter_trimming_enabled);

    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    BarcodingInfo::FilterSet allowed_barcodes;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, false);
        allowed_barcodes = sample_sheet->get_barcode_values();
    }

    SamHdrPtr hdr(sam_hdr_init());
    cli::add_pg_hdr(hdr.get(), args, device);

    if (barcode_enabled) {
        std::unordered_map<std::string, std::string> custom_barcodes{};
        if (custom_barcode_file) {
            custom_barcodes = demux::parse_custom_sequences(*custom_barcode_file);
        }
        if (custom_kit) {
            auto [kit_name, kit_info] = get_custom_barcode_kit_info(*custom_kit);
            utils::add_rg_headers_with_barcode_kit(hdr.get(), read_groups, kit_name, kit_info,
                                                   custom_barcodes, sample_sheet.get());
        } else {
            const auto& kit_info = get_barcode_kit_info(barcode_kit);
            utils::add_rg_headers_with_barcode_kit(hdr.get(), read_groups, barcode_kit, kit_info,
                                                   custom_barcodes, sample_sheet.get());
        }
    } else {
        utils::add_rg_headers(hdr.get(), read_groups);
    }

    utils::HtsFile hts_file("-", output_mode, thread_allocations.writer_threads, false);

    PipelineDescriptor pipeline_desc;
    std::string gpu_names{};
#if DORADO_CUDA_BUILD
    gpu_names = utils::get_cuda_gpu_names(device);
#endif
    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, hts_file, gpu_names);
    auto aligner = PipelineDescriptor::InvalidNodeHandle;
    auto current_sink_node = hts_writer;
    if (enable_aligner) {
        auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
        aligner = pipeline_desc.add_node<AlignerNode>({current_sink_node}, index_file_access, ref,
                                                      "", aligner_options,
                                                      thread_allocations.aligner_threads);
        current_sink_node = aligner;
    }
    current_sink_node = pipeline_desc.add_node<ReadToBamTypeNode>(
            {current_sink_node}, emit_moves, thread_allocations.read_converter_threads,
            methylation_threshold_pct, std::move(sample_sheet), 1000);
    if (estimate_poly_a) {
        current_sink_node = pipeline_desc.add_node<PolyACalculatorNode>(
                {current_sink_node}, std::thread::hardware_concurrency(), 1000);
    }
    if (adapter_trimming_enabled) {
        current_sink_node = pipeline_desc.add_node<AdapterDetectorNode>(
                {current_sink_node}, thread_allocations.adapter_threads, !adapter_no_trim,
                !primer_no_trim, custom_primer_file);
    }
    if (barcode_enabled) {
        std::vector<std::string> kit_as_vector{barcode_kit};
        current_sink_node = pipeline_desc.add_node<BarcodeClassifierNode>(
                {current_sink_node}, thread_allocations.barcoder_threads, kit_as_vector,
                barcode_both_ends, barcode_no_trim, std::move(allowed_barcodes), custom_kit,
                custom_barcode_file);
    }
    current_sink_node = pipeline_desc.add_node<ReadFilterNode>(
            {current_sink_node}, min_qscore, default_parameters.min_sequence_length,
            std::unordered_set<std::string>{}, thread_allocations.read_filter_threads);

    auto mean_qscore_start_pos = model_config.mean_qscore_start_pos;

    api::create_simplex_pipeline(
            pipeline_desc, std::move(runners), std::move(remora_runners), overlap,
            mean_qscore_start_pos, !adapter_no_trim, thread_allocations.scaler_node_threads,
            true /* Enable read splitting */, thread_allocations.splitter_node_threads,
            thread_allocations.remora_threads, current_sink_node,
            PipelineDescriptor::InvalidNodeHandle);

    // Create the Pipeline from our description.
    std::vector<dorado::stats::StatsReporter> stats_reporters{dorado::stats::sys_stats_report};
    auto pipeline = Pipeline::create(std::move(pipeline_desc), &stats_reporters);
    if (pipeline == nullptr) {
        spdlog::error("Failed to create pipeline");
        std::exit(EXIT_FAILURE);
    }

    // At present, header output file header writing relies on direct node method calls
    // rather than the pipeline framework.
    auto& hts_writer_ref = dynamic_cast<HtsWriter&>(pipeline->get_node_ref(hts_writer));
    if (enable_aligner) {
        const auto& aligner_ref = dynamic_cast<AlignerNode&>(pipeline->get_node_ref(aligner));
        utils::add_sq_hdr(hdr.get(), aligner_ref.get_sequence_records_for_header());
    }
    hts_file.set_header(hdr.get());

    std::unordered_set<std::string> reads_already_processed;
    if (!resume_from_file.empty()) {
        spdlog::info("> Inspecting resume file...");
        // Turn off warning logging as header info is fetched.
        auto initial_hts_log_level = hts_get_log_level();
        hts_set_log_level(HTS_LOG_OFF);
        auto pg_keys = utils::extract_pg_keys_from_hdr(resume_from_file, {"CL"});
        hts_set_log_level(initial_hts_log_level);

        auto tokens = cli::extract_token_from_cli(pg_keys["CL"]);
        // First token is the dorado binary name. Remove that because the
        // sub parser only knows about the `basecaller` command.
        tokens.erase(tokens.begin());
        resume_parser.parse_args(tokens);

        const std::string model_arg = resume_parser.get<std::string>("model");
        const ModelSelection resume_selection = ModelComplexParser::parse(model_arg);

        if (resume_selection.is_path()) {
            // If the model selection is a path, check it exists and matches
            const auto resume_model_name =
                    models::extract_model_name_from_path(fs::path(model_arg));
            if (model_name != resume_model_name) {
                throw std::runtime_error(
                        "Resume only works if the same model is used. Resume model was " +
                        resume_model_name + " and current model is " + model_name);
            }
        } else if (resume_selection != model_selection) {
            throw std::runtime_error(
                    "Resume only works if the same model is used. Resume model complex was " +
                    resume_selection.raw + " and current model is " + model_selection.raw);
        }

        // Resume functionality injects reads directly into the writer node.
        ResumeLoaderNode resume_loader(hts_writer_ref, resume_from_file);
        resume_loader.copy_completed_reads();
        reads_already_processed = resume_loader.get_processed_read_ids();
    }

    // If we're doing alignment, post-processing takes longer due to bam file sorting.
    float post_processing_percentage = (hts_file.finalise_is_noop() || ref.empty()) ? 0.0f : 0.5f;

    ProgressTracker tracker(int(num_reads), false, post_processing_percentage);
    tracker.set_description("Basecalling");

    std::vector<dorado::stats::StatsCallable> stats_callables;
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

    DataLoader loader(*pipeline, "cpu", thread_allocations.loader_threads, max_reads, read_list,
                      reads_already_processed);

    DefaultClientInfo::PolyTailSettings polytail_settings{estimate_poly_a,
                                                          is_rna_model(model_config), polya_config};
    auto default_client_info = std::make_shared<DefaultClientInfo>(polytail_settings);
    auto func = [default_client_info](ReadCommon& read) { read.client_info = default_client_info; };
    loader.add_read_initialiser(func);

    // Run pipeline.
    loader.load_reads(data_path, recursive_file_loading, ReadOrder::UNRESTRICTED);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    // Stop the stats sampler thread before tearing down any pipeline objects.
    // Then update progress tracking one more time from this thread, to
    // allow accurate summarisation.
    stats_sampler->terminate();
    tracker.update_progress_bar(final_stats);

    // Report progress during output file finalisation.
    tracker.set_description("Sorting output files");
    hts_file.finalise([&](size_t progress) {
        tracker.update_post_processing_progress(static_cast<float>(progress));
    });

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

int basecaller(int argc, char* argv[]) {
    utils::set_torch_allocator_max_split_size();
    utils::make_torch_deterministic();
    torch::set_num_threads(1);

    cli::ArgParser parser("dorado");

    parser.visible.add_argument("model").help(
            "model selection {fast,hac,sup}@v{version} for automatic model selection including "
            "modbases, or path to existing model directory");

    parser.visible.add_argument("data").help("the data directory or file (POD5/FAST5 format).");

    int verbosity = 0;
    parser.visible.add_argument("-v", "--verbose")
            .default_value(false)
            .implicit_value(true)
            .nargs(0)
            .action([&](const auto&) { ++verbosity; })
            .append();

    parser.visible.add_argument("-x", "--device")
            .help("device string in format \"cuda:0,...,N\", \"cuda:all\", \"metal\", \"cpu\" "
                  "etc..")
            .default_value(default_parameters.device);

    parser.visible.add_argument("-l", "--read-ids")
            .help("A file with a newline-delimited list of reads to basecall. If not provided, all "
                  "reads will be basecalled")
            .default_value(std::string(""));

    parser.visible.add_argument("--resume-from")
            .help("Resume basecalling from the given HTS file. Fully written read records are not "
                  "processed again.")
            .default_value(std::string(""));

    parser.visible.add_argument("-n", "--max-reads").default_value(0).scan<'i', int>();

    parser.visible.add_argument("--min-qscore")
            .help("Discard reads with mean Q-score below this threshold.")
            .default_value(0)
            .scan<'i', int>();

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

    parser.visible.add_argument("--modified-bases")
            .nargs(argparse::nargs_pattern::at_least_one)
            .action([](const std::string& value) {
                const auto& mods = models::modified_model_variants();
                if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                    spdlog::error("'{}' is not a supported modification please select from {}",
                                  value,
                                  std::accumulate(std::next(mods.begin()), mods.end(), mods[0],
                                                  [](std::string const& a, std::string const& b) {
                                                      return a + ", " + b;
                                                  }));
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

    parser.visible.add_argument("--emit-fastq")
            .help("Output in fastq format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--emit-sam")
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);

    parser.visible.add_argument("--emit-moves").default_value(false).implicit_value(true);

    parser.visible.add_argument("--reference")
            .help("Path to reference for alignment.")
            .default_value(std::string(""));

    parser.visible.add_argument("--kit-name")
            .help("Enable barcoding with the provided kit name. Choose from: " +
                  dorado::barcode_kits::barcode_kits_list_str() + ".")
            .default_value(std::string{});
    parser.visible.add_argument("--barcode-both-ends")
            .help("Require both ends of a read to be barcoded for a double ended barcode.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--no-trim")
            .help("Skip trimming of barcodes, adapters, and primers. If option is not chosen, "
                  "trimming of all three is enabled.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--trim")
            .help("Specify what to trim. Options are 'none', 'all', 'adapters', and 'primers'. "
                  "Default behaviour is to trim all detected adapters, primers, or barcodes. "
                  "Choose 'adapters' to just trim adapters. The 'primers' choice will trim "
                  "adapters and "
                  "primers, but not barcodes. The 'none' choice is equivelent to using --no-trim. "
                  "Note that "
                  "this only applies to DNA. RNA adapters are always trimmed.")
            .default_value(std::string(""));
    parser.visible.add_argument("--sample-sheet")
            .help("Path to the sample sheet to use.")
            .default_value(std::string(""));
    parser.visible.add_argument("--barcode-arrangement")
            .help("Path to file with custom barcode arrangement.")
            .default_value(std::nullopt);
    parser.visible.add_argument("--barcode-sequences")
            .help("Path to file with custom barcode sequences.")
            .default_value(std::nullopt);
    parser.visible.add_argument("--primer-sequences")
            .help("Path to file with custom primer sequences.")
            .default_value(std::nullopt);
    parser.visible.add_argument("--estimate-poly-a")
            .help("Estimate poly-A/T tail lengths (beta feature). Primarily meant for cDNA and "
                  "dRNA use cases. Note that if this flag is set, then adapter/primer/barcode "
                  "trimming will be disabled.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--poly-a-config")
            .help("Configuration file for PolyA estimation to change default behaviours")
            .default_value(std::string(""));

    cli::add_minimap2_arguments(parser, alignment::dflt_options);
    cli::add_internal_arguments(parser);

    // Create a copy of the parser to use if the resume feature is enabled. Needed
    // to parse the model used for the file being resumed from. Note that this copy
    // needs to be made __before__ the parser is used.
    auto resume_parser = parser.visible;

    try {
        cli::parse(parser, argc, argv);
    } catch (const std::exception& e) {
        std::ostringstream parser_stream;
        parser_stream << parser.visible;
        spdlog::error("{}\n{}", e.what(), parser_stream.str());
        return EXIT_FAILURE;
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.visible.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    const auto model_arg = parser.visible.get<std::string>("model");
    const auto data = parser.visible.get<std::string>("data");
    const auto recursive = parser.visible.get<bool>("--recursive");
    const auto mod_bases = parser.visible.get<std::vector<std::string>>("--modified-bases");
    const auto mod_bases_models = parser.visible.get<std::string>("--modified-bases-models");

    const ModelSelection model_selection = cli::parse_model_argument(model_arg);

    auto methylation_threshold = parser.visible.get<float>("--modified-bases-threshold");
    if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
        spdlog::error("--modified-bases-threshold must be between 0 and 1.");
        return EXIT_FAILURE;
    }

    auto output_mode = OutputMode::BAM;

    auto emit_fastq = parser.visible.get<bool>("--emit-fastq");
    auto emit_sam = parser.visible.get<bool>("--emit-sam");

    if (emit_fastq && emit_sam) {
        spdlog::error("Only one of --emit-{fastq, sam} can be set (or none).");
        return EXIT_FAILURE;
    }

    if (emit_fastq) {
        if (model_selection.has_mods_variant() || !mod_bases.empty() || !mod_bases_models.empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with modbase models as FASTQ cannot store modbase "
                    "results.");
            return EXIT_FAILURE;
        }
        if (!parser.visible.get<std::string>("--reference").empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with --reference as FASTQ cannot store alignment "
                    "results.");
            return EXIT_FAILURE;
        }
        spdlog::info(" - Note: FASTQ output is not recommended as not all data can be preserved.");
        output_mode = OutputMode::FASTQ;
    } else if (emit_sam || utils::is_fd_tty(stdout)) {
        output_mode = OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = OutputMode::UBAM;
    }

    bool no_trim_barcodes = false, no_trim_primers = false, no_trim_adapters = false;
    auto trim_options = parser.visible.get<std::string>("--trim");
    if (parser.visible.get<bool>("--no-trim")) {
        if (!trim_options.empty()) {
            spdlog::error("Only one of --no-trim and --trim can be used.");
            return EXIT_FAILURE;
        }
        no_trim_barcodes = no_trim_primers = no_trim_adapters = true;
    }
    if (trim_options == "none") {
        no_trim_barcodes = no_trim_primers = no_trim_adapters = true;
    } else if (trim_options == "primers") {
        no_trim_barcodes = true;
    } else if (trim_options == "adapters") {
        no_trim_barcodes = no_trim_primers = true;
    } else if (!trim_options.empty() && trim_options != "all") {
        spdlog::error("Unsupported --trim value '{}'.", trim_options);
        return EXIT_FAILURE;
    }

    std::string polya_config = "";
    if (parser.visible.get<bool>("--estimate-poly-a")) {
        if (trim_options == "primers" || trim_options == "adapters" || trim_options == "all") {
            spdlog::error(
                    "--trim cannot be used with options 'primers', 'adapters', or 'all', "
                    "if you are also using --estimate-poly-a.");
            return EXIT_FAILURE;
        }
        no_trim_primers = no_trim_adapters = no_trim_barcodes = true;
        spdlog::info(
                "Estimation of poly-a has been requested, so adapter/primer/barcode trimming has "
                "been disabled.");
        polya_config = parser.visible.get<std::string>("--poly-a-config");
    }

    if (parser.visible.is_used("--kit-name") && parser.visible.is_used("--barcode-arrangement")) {
        spdlog::error(
                "--kit-name and --barcode-arrangement cannot be used together. Please provide only "
                "one.");
        return EXIT_FAILURE;
    }

    std::optional<std::string> custom_kit = std::nullopt;
    if (parser.visible.is_used("--barcode-arrangement")) {
        custom_kit = parser.visible.get<std::string>("--barcode-arrangement");
    }

    std::optional<std::string> custom_barcode_seqs = std::nullopt;
    if (parser.visible.is_used("--barcode-sequences")) {
        custom_barcode_seqs = parser.visible.get<std::string>("--barcode-sequences");
    }

    std::optional<std::string> custom_primer_file = std::nullopt;
    if (parser.visible.is_used("--primer-sequences")) {
        custom_primer_file = parser.visible.get<std::string>("--primer-sequences");
    }

    // Assert that only one of --modified-bases, --modified-bases-models or mods model complex is set
    auto ways = {model_selection.has_mods_variant(), !mod_bases.empty(), !mod_bases_models.empty()};
    if (std::count(ways.begin(), ways.end(), true) > 1) {
        spdlog::error(
                "Only one of --modified-bases, --modified-bases-models, or modified models set "
                "via models argument can be used at once");
        return EXIT_FAILURE;
    };

    fs::path model_path;
    std::vector<fs::path> mods_model_paths;
    std::set<fs::path> temp_download_paths;

    if (model_selection.is_path()) {
        model_path = fs::path(model_arg);
        mods_model_paths =
                dorado::get_non_complex_mods_models(model_path, mod_bases, mod_bases_models);
    } else {
        auto model_finder = cli::model_finder(model_selection, data, recursive, true);
        try {
            model_path = model_finder.fetch_simplex_model();
            if (model_selection.has_mods_variant()) {
                // Get mods models from complex - we assert above that there's only one method
                mods_model_paths = model_finder.fetch_mods_models();
            } else {
                // Get mods models from args
                mods_model_paths = dorado::get_non_complex_mods_models(model_path, mod_bases,
                                                                       mod_bases_models);
            }
            temp_download_paths = model_finder.downloaded_models();
        } catch (std::exception& e) {
            spdlog::error(e.what());
            utils::clean_temporary_models(model_finder.downloaded_models());
            return EXIT_FAILURE;
        }
    }

    spdlog::info("> Creating basecall pipeline");

    try {
        setup(args, model_path, data, mods_model_paths, parser.visible.get<std::string>("-x"),
              parser.visible.get<std::string>("--reference"), parser.visible.get<int>("-c"),
              parser.visible.get<int>("-o"), parser.visible.get<int>("-b"),
              default_parameters.num_runners, default_parameters.remora_batchsize,
              default_parameters.remora_threads, methylation_threshold, output_mode,
              parser.visible.get<bool>("--emit-moves"), parser.visible.get<int>("--max-reads"),
              parser.visible.get<int>("--min-qscore"),
              parser.visible.get<std::string>("--read-ids"), recursive,
              cli::process_minimap2_arguments(parser, alignment::dflt_options),
              parser.hidden.get<bool>("--skip-model-compatibility-check"),
              parser.hidden.get<std::string>("--dump_stats_file"),
              parser.hidden.get<std::string>("--dump_stats_filter"),
              parser.visible.get<std::string>("--resume-from"),
              parser.visible.get<std::string>("--kit-name"),
              parser.visible.get<bool>("--barcode-both-ends"), no_trim_barcodes, no_trim_adapters,
              no_trim_primers, parser.visible.get<std::string>("--sample-sheet"), custom_kit,
              custom_barcode_seqs, custom_primer_file, resume_parser,
              parser.visible.get<bool>("--estimate-poly-a"), polya_config, model_selection);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        utils::clean_temporary_models(temp_download_paths);
        return EXIT_FAILURE;
    }

    utils::clean_temporary_models(temp_download_paths);
    spdlog::info("> Finished");
    return EXIT_SUCCESS;
}

}  // namespace dorado
