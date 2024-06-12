#include "basecall_output_args.h"

#include "utils/tty_utils.h"

#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>

using OutputMode = dorado::utils::HtsFile::OutputMode;
namespace fs = std::filesystem;

namespace dorado::cli {

namespace {

constexpr std::string_view OUTPUT_DIR_ARG{"--output-dir"};
constexpr std::string_view EMIT_FASTQ_ARG{"--emit-fastq"};
constexpr std::string_view EMIT_SAM_ARG{"--emit-sam"};

constexpr std::string_view OUTPUT_FILE_PREFIX{"calls_"};
constexpr std::string_view FASTQ_EXT{".fastq"};
constexpr std::string_view SAM_EXT{".sam"};
constexpr std::string_view BAM_EXT{".bam"};

bool try_create_output_folder(const std::string& output_folder) {
    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    fs::create_directories(fs::path{output_folder}, creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder {}. ErrorCode({}) {}", output_folder,
                      creation_error.value(), creation_error.message());
        return false;
    }
    return true;
}

std::tm get_gmtime(const std::time_t* time) {
    // gmtime is not threadsafe, so lock.
    static std::mutex gmtime_mutex;
    std::lock_guard lock(gmtime_mutex);
    std::tm* time_buffer = gmtime(time);
    return *time_buffer;
}

class HtsFileCreator {
    const bool m_emit_fastq;
    const bool m_emit_sam;
    const bool m_reference_requested;
    std::optional<std::string> m_output_dir;

    std::string m_output_file{};
    OutputMode m_output_mode{OutputMode::BAM};

    bool is_output_to_file() const { return m_output_file != "-"; }

    bool is_sam_or_bam_output() const {
        return m_output_mode == OutputMode::BAM || m_output_mode == OutputMode::SAM;
    }

    std::string_view get_output_file_extension() {
        if (m_emit_fastq) {
            return FASTQ_EXT;
        }
        if (m_emit_sam) {
            return SAM_EXT;
        }
        return BAM_EXT;
    }

    std::string get_output_filename() {
        time_t time_now = time(nullptr);
        std::tm gm_time_now = get_gmtime(&time_now);
        char timestamp_buffer[32];
        strftime(timestamp_buffer, 32, "%F_T%H-%M-%S", &gm_time_now);

        std::ostringstream oss{};
        oss << OUTPUT_FILE_PREFIX << timestamp_buffer << get_output_file_extension();
        return oss.str();
    }

    bool try_set_output_file() {
        if (!m_output_dir) {
            m_output_file = "-";
            return true;
        }
        if (!try_create_output_folder(*m_output_dir)) {
            return false;
        }

        m_output_file = (fs::path(*m_output_dir) / get_output_filename()).string();
        return true;
    }

    bool try_set_output_mode() {
        if (m_emit_fastq) {
            if (m_emit_sam) {
                spdlog::error("Only one of --emit-{fastq, sam} can be set (or none).");
                return false;
            }
            if (m_reference_requested) {
                spdlog::error(
                        "--emit-fastq cannot be used with --reference as FASTQ cannot store "
                        "alignment results.");
                return false;
            }
            spdlog::info(
                    " - Note: FASTQ output is not recommended as not all data can be preserved.");
            m_output_mode = OutputMode::FASTQ;
        } else if (m_emit_sam || (m_output_file == "-" && utils::is_fd_tty(stdout))) {
            m_output_mode = OutputMode::SAM;
        } else if (m_output_file == "-" && utils::is_fd_pipe(stdout)) {
            m_output_mode = OutputMode::UBAM;
        } else {
            m_output_mode = OutputMode::BAM;
        }

        return true;
    }

public:
    HtsFileCreator(const utils::arg_parse::ArgParser& parser)
            : m_emit_fastq(parser.visible.get<bool>(EMIT_FASTQ_ARG)),
              m_emit_sam(parser.visible.get<bool>(EMIT_SAM_ARG)),
              m_reference_requested(!parser.visible.get<std::string>("--reference").empty()),
              m_output_dir(parser.visible.present<std::string>(OUTPUT_DIR_ARG)) {}

    std::unique_ptr<utils::HtsFile> create() {
        if (!try_set_output_file()) {
            return nullptr;
        }
        if (!try_set_output_mode()) {
            return nullptr;
        }

        // If writing to a SAM/BAM file and there's a reference then we will sort.
        auto sort_bam = is_output_to_file() && is_sam_or_bam_output() && m_reference_requested;

        return std::make_unique<utils::HtsFile>(m_output_file, m_output_mode, 0, sort_bam);
    }
};

}  // namespace

std::unique_ptr<utils::HtsFile> extract_hts_file(const utils::arg_parse::ArgParser& parser) {
    HtsFileCreator hts_file_creator(parser);
    return hts_file_creator.create();
}

void add_basecaller_output_arguments(utils::arg_parse::ArgParser& parser) {
    parser.visible.add_argument(EMIT_FASTQ_ARG)
            .help("Output in fastq format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument(EMIT_SAM_ARG)
            .help("Output in SAM format.")
            .default_value(false)
            .implicit_value(true);
    parser.visible.add_argument("-o", OUTPUT_DIR_ARG)
            .help("Optional output folder, if specified output will be written to a calls file "
                  "(calls_<timestamp>.sam|.bam|.fastq) in the given folder.");
}

}  // namespace dorado::cli