#include "Version.h"
#include "cli/cli_utils.h"
#include "data_loader/DataLoader.h"
#include "models/models.h"
#include "nn/CRFModelConfig.h"
#include "nn/ModBaseRunner.h"
#include "nn/Runners.h"
#include "read_pipeline/AlignerNode.h"
#include "read_pipeline/BarcodeClassifierNode.h"
#include "read_pipeline/HtsReader.h"
#include "read_pipeline/HtsWriter.h"
#include "read_pipeline/Pipelines.h"
#include "read_pipeline/PolyACalculator.h"
#include "read_pipeline/ProgressTracker.h"
#include "read_pipeline/ReadFilterNode.h"
#include "read_pipeline/ReadToBamTypeNode.h"
#include "read_pipeline/ResumeLoaderNode.h"
#include "utils/SampleSheet.h"
#include "utils/bam_utils.h"
#include "utils/barcode_kits.h"
#include "utils/basecaller_utils.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/stats.h"
#include "utils/sys_stats.h"
#include "utils/torch_utils.h"

#include <argparse.hpp>
#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

namespace dorado {

using dorado::utils::default_parameters;
using namespace std::chrono_literals;

void setup(std::vector<std::string> args,
           const std::filesystem::path& model_path,
           const std::string& data_path,
           const std::string& remora_models,
           const std::string& device,
           const std::string& ref,
           size_t chunk_size,
           size_t overlap,
           size_t batch_size,
           size_t num_runners,
           size_t remora_batch_size,
           size_t num_remora_threads,
           float methylation_threshold_pct,
           HtsWriter::OutputMode output_mode,
           bool emit_moves,
           size_t max_reads,
           size_t min_qscore,
           std::string read_list_file_path,
           bool recursive_file_loading,
           const alignment::Minimap2Options& aligner_options,
           bool skip_model_compatibility_check,
           const std::string& dump_stats_file,
           const std::string& dump_stats_filter,
           const std::string& resume_from_file,
           const std::vector<std::string>& barcode_kits,
           bool barcode_both_ends,
           bool barcode_no_trim,
           const std::string& barcode_sample_sheet,
           argparse::ArgumentParser& resume_parser,
           bool estimate_poly_a) {
    auto model_config = load_crf_model_config(model_path);
    std::string model_name = models::extract_model_from_model_path(model_path.string());

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

    // Check sample rate of model vs data.
    auto data_sample_rate = DataLoader::get_sample_rate(data_path, recursive_file_loading);
    auto model_sample_rate = model_config.sample_rate;
    if (model_sample_rate < 0) {
        // If unsuccessful, find sample rate by model name.
        model_sample_rate = models::get_sample_rate_by_model_name(model_name);
    }
    if (!skip_model_compatibility_check &&
        !sample_rates_compatible(data_sample_rate, model_sample_rate)) {
        std::stringstream err;
        err << "Sample rate for model (" << model_sample_rate << ") and data (" << data_sample_rate
            << ") are not compatible.";
        throw std::runtime_error(err.str());
    }

    const bool enable_aligner = !ref.empty();

    // create modbase runners first so basecall runners can pick batch sizes based on available memory
    auto remora_runners = create_modbase_runners(
            remora_models, device, default_parameters.remora_runners_per_caller, remora_batch_size);

    auto [runners, num_devices] =
            create_basecall_runners(model_config, device, num_runners, 0, batch_size, chunk_size);

    auto read_groups = DataLoader::load_read_groups(data_path, model_name, recursive_file_loading);

    bool duplex = false;

    const auto thread_allocations = utils::default_thread_allocations(
            int(num_devices), !remora_runners.empty() ? int(num_remora_threads) : 0, enable_aligner,
            !barcode_kits.empty());

    std::unique_ptr<const utils::SampleSheet> sample_sheet;
    BarcodingInfo::FilterSet allowed_barcodes;
    if (!barcode_sample_sheet.empty()) {
        sample_sheet = std::make_unique<const utils::SampleSheet>(barcode_sample_sheet, false);
        allowed_barcodes = sample_sheet->get_barcode_values();
    }

    SamHdrPtr hdr(sam_hdr_init());
    cli::add_pg_hdr(hdr.get(), args);
    utils::add_rg_hdr(hdr.get(), read_groups, barcode_kits, sample_sheet.get());

    PipelineDescriptor pipeline_desc;
    auto hts_writer = pipeline_desc.add_node<HtsWriter>(
            {}, "-", output_mode, thread_allocations.writer_threads, num_reads);
    auto aligner = PipelineDescriptor::InvalidNodeHandle;
    auto current_sink_node = hts_writer;
    if (enable_aligner) {
        auto index_file_access = std::make_shared<alignment::IndexFileAccess>();
        aligner = pipeline_desc.add_node<AlignerNode>({current_sink_node}, index_file_access, ref,
                                                      aligner_options,
                                                      thread_allocations.aligner_threads);
        current_sink_node = aligner;
    }
    current_sink_node = pipeline_desc.add_node<ReadToBamType>(
            {current_sink_node}, emit_moves, thread_allocations.read_converter_threads,
            methylation_threshold_pct, std::move(sample_sheet), 1000);
    if (estimate_poly_a) {
        current_sink_node = pipeline_desc.add_node<PolyACalculator>(
                {current_sink_node}, std::thread::hardware_concurrency(),
                is_rna_model(model_config));
    }
    if (!barcode_kits.empty()) {
        current_sink_node = pipeline_desc.add_node<BarcodeClassifierNode>(
                {current_sink_node}, thread_allocations.barcoder_threads, barcode_kits,
                barcode_both_ends, barcode_no_trim, std::move(allowed_barcodes));
    }
    current_sink_node = pipeline_desc.add_node<ReadFilterNode>(
            {current_sink_node}, min_qscore, default_parameters.min_sequence_length,
            std::unordered_set<std::string>{}, thread_allocations.read_filter_threads);

    auto mean_qscore_start_pos = model_config.mean_qscore_start_pos;
    if (mean_qscore_start_pos < 0) {
        mean_qscore_start_pos = models::get_mean_qscore_start_pos_by_model_name(model_name);
        if (mean_qscore_start_pos < 0) {
            throw std::runtime_error("Mean q-score start position cannot be < 0");
        }
    }
    pipelines::create_simplex_pipeline(
            pipeline_desc, std::move(runners), std::move(remora_runners), overlap,
            mean_qscore_start_pos, thread_allocations.scaler_node_threads,
            true /* Enable read splitting */, thread_allocations.splitter_node_threads,
            int(thread_allocations.remora_threads * num_devices), current_sink_node);

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
    hts_writer_ref.set_and_write_header(hdr.get());

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
        auto resume_model_name =
                models::extract_model_from_model_path(resume_parser.get<std::string>("model"));
        if (model_name != resume_model_name) {
            throw std::runtime_error(
                    "Resume only works if the same model is used. Resume model was " +
                    resume_model_name + " and current model is " + model_name);
        }
        // Resume functionality injects reads directly into the writer node.
        ResumeLoaderNode resume_loader(hts_writer_ref, resume_from_file);
        resume_loader.copy_completed_reads();
        reads_already_processed = resume_loader.get_processed_read_ids();
    }

    std::vector<dorado::stats::StatsCallable> stats_callables;
    ProgressTracker tracker(int(num_reads), duplex);
    stats_callables.push_back(
            [&tracker](const stats::NamedStats& stats) { tracker.update_progress_bar(stats); });
    constexpr auto kStatsPeriod = 100ms;
    const size_t max_stats_records = static_cast<size_t>(dump_stats_file.empty() ? 0 : 100000);
    auto stats_sampler = std::make_unique<dorado::stats::StatsSampler>(
            kStatsPeriod, stats_reporters, stats_callables, max_stats_records);

    DataLoader loader(*pipeline, "cpu", thread_allocations.loader_threads, max_reads, read_list,
                      reads_already_processed);

    // Run pipeline.
    loader.load_reads(data_path, recursive_file_loading);

    // Wait for the pipeline to complete.  When it does, we collect
    // final stats to allow accurate summarisation.
    auto final_stats = pipeline->terminate(DefaultFlushOptions());

    // Stop the stats sampler thread before tearing down any pipeline objects.
    stats_sampler->terminate();

    // Then update progress tracking one more time from this thread, to
    // allow accurate summarisation.
    tracker.update_progress_bar(final_stats);
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
    utils::InitLogging();
    utils::make_torch_deterministic();
    torch::set_num_threads(1);

    cli::ArgParser parser("dorado");

    parser.visible.add_argument("model").help("the basecaller model to run.");

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
                const auto& mods = models::modified_mods();
                if (std::find(mods.begin(), mods.end(), value) == mods.end()) {
                    spdlog::error(
                            "'{}' is not a supported modification please select from {}", value,
                            std::accumulate(
                                    std::next(mods.begin()), mods.end(), mods[0],
                                    [](std::string a, std::string b) { return a + ", " + b; }));
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
                  dorado::barcode_kits::barcode_kits_list_str() + ".");
    parser.visible.add_argument("--barcode-both-ends")
            .help("Require both ends of a read to be barcoded for a double ended barcode.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--no-trim")
            .help("Skip barcode trimming. If option is not chosen, trimming is enabled.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("--sample-sheet")
            .help("Path to the sample sheet to use.")
            .default_value(std::string(""));
    parser.visible.add_argument("--estimate-poly-a")
            .help("Estimate poly-A/T tail lengths (beta feature). Primarily meant "
                  "for cDNA and "
                  "dRNA use cases.")
            .default_value(false)
            .implicit_value(true);

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
        std::exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    if (parser.visible.get<bool>("--verbose")) {
        utils::SetVerboseLogging(static_cast<dorado::utils::VerboseLogLevel>(verbosity));
    }

    auto model = parser.visible.get<std::string>("model");
    auto mod_bases = parser.visible.get<std::vector<std::string>>("--modified-bases");
    auto mod_bases_models = parser.visible.get<std::string>("--modified-bases-models");

    if (!mod_bases.empty() && !mod_bases_models.empty()) {
        spdlog::error(
                "only one of --modified-bases or --modified-bases-models should be specified.");
        std::exit(EXIT_FAILURE);
    } else if (mod_bases.size()) {
        std::vector<std::string> m;
        std::transform(
                mod_bases.begin(), mod_bases.end(), std::back_inserter(m),
                [&model](std::string m) { return models::get_modification_model(model, m); });

        mod_bases_models =
                std::accumulate(std::next(m.begin()), m.end(), m[0],
                                [](std::string a, std::string b) { return a + "," + b; });
    }

    auto methylation_threshold = parser.visible.get<float>("--modified-bases-threshold");
    if (methylation_threshold < 0.f || methylation_threshold > 1.f) {
        spdlog::error("--modified-bases-threshold must be between 0 and 1.");
        std::exit(EXIT_FAILURE);
    }

    auto output_mode = HtsWriter::OutputMode::BAM;

    auto emit_fastq = parser.visible.get<bool>("--emit-fastq");
    auto emit_sam = parser.visible.get<bool>("--emit-sam");

    if (emit_fastq && emit_sam) {
        throw std::runtime_error("Only one of --emit-{fastq, sam} can be set (or none).");
    }

    if (emit_fastq) {
        if (!mod_bases.empty() || !mod_bases_models.empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with modbase models as FASTQ cannot store modbase "
                    "results.");
            std::exit(EXIT_FAILURE);
        }
        if (!parser.visible.get<std::string>("--reference").empty()) {
            spdlog::error(
                    "--emit-fastq cannot be used with --reference as FASTQ cannot store alignment "
                    "results.");
            std::exit(EXIT_FAILURE);
        }
        spdlog::info(" - Note: FASTQ output is not recommended as not all data can be preserved.");
        output_mode = HtsWriter::OutputMode::FASTQ;
    } else if (emit_sam || utils::is_fd_tty(stdout)) {
        output_mode = HtsWriter::OutputMode::SAM;
    } else if (utils::is_fd_pipe(stdout)) {
        output_mode = HtsWriter::OutputMode::UBAM;
    }

    spdlog::info("> Creating basecall pipeline");

    try {
        setup(args, model, parser.visible.get<std::string>("data"), mod_bases_models,
              parser.visible.get<std::string>("-x"), parser.visible.get<std::string>("--reference"),
              parser.visible.get<int>("-c"), parser.visible.get<int>("-o"),
              parser.visible.get<int>("-b"), default_parameters.num_runners,
              default_parameters.remora_batchsize, default_parameters.remora_threads,
              methylation_threshold, output_mode, parser.visible.get<bool>("--emit-moves"),
              parser.visible.get<int>("--max-reads"), parser.visible.get<int>("--min-qscore"),
              parser.visible.get<std::string>("--read-ids"),
              parser.visible.get<bool>("--recursive"),
              cli::process_minimap2_arguments(parser, alignment::dflt_options),
              parser.hidden.get<bool>("--skip-model-compatibility-check"),
              parser.hidden.get<std::string>("--dump_stats_file"),
              parser.hidden.get<std::string>("--dump_stats_filter"),
              parser.visible.get<std::string>("--resume-from"),
              parser.visible.get<std::vector<std::string>>("--kit-name"),
              parser.visible.get<bool>("--barcode-both-ends"),
              parser.visible.get<bool>("--no-trim"),
              parser.visible.get<std::string>("--sample-sheet"), resume_parser,
              parser.visible.get<bool>("--estimate-poly-a"));
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }

    spdlog::info("> Finished");
    return 0;
}

}  // namespace dorado
