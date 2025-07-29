#include "hts_writer/Structure.h"

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "utils/SampleSheet.h"
#include "utils/barcode_kits.h"
#include "utils/time_utils.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dorado {
namespace hts_writer {

namespace fs = std::filesystem;

namespace {
std::tm get_gmtime(const std::time_t* time) {
    // gmtime is not threadsafe, so lock.
    static std::mutex gmtime_mutex;
    std::lock_guard lock(gmtime_mutex);
    std::tm* time_buffer = gmtime(time);
    return *time_buffer;
}

void create_output_folder(const std::filesystem::path& path) {
#ifdef _WIN32
    static std::once_flag long_path_warning_flag;
    if (path.string().size() >= 260) {
        std::call_once(long_path_warning_flag, [&path] {
            spdlog::warn("Filepaths longer than 260 characters may cause issues on Windows.");
            spdlog::warn("Attempting to create path of length '{}' at '{}'.", path.string().size(),
                         path.string());
        });
    }
#endif

    std::error_code creation_error;
    // N.B. No error code if folder already exists.
    fs::create_directories(path.parent_path(), creation_error);
    if (creation_error) {
        spdlog::error("Unable to create output folder '{}'.  ErrorCode({}) {}", path.string(),
                      creation_error.value(), creation_error.message());
        throw std::runtime_error("Failed to create output directory");
    }
}

}  // namespace

SingleFileStructure::SingleFileStructure(const std::string& output_dir, OutputMode mode)
        : m_mode(mode), m_path((fs::path(output_dir) / get_filename())) {
    create_output_folder(m_path);
};

std::string SingleFileStructure::get_path([[maybe_unused]] const HtsData& hts_data) {
    return m_path.string();
};

constexpr std::string_view OUTPUT_FILE_PREFIX{"calls_"};

std::string SingleFileStructure::get_filename() const {
    time_t time_now = time(nullptr);
    std::tm gm_time_now = get_gmtime(&time_now);
    char timestamp_buffer[32];
    strftime(timestamp_buffer, 32, "%F_T%H-%M-%S", &gm_time_now);

    std::ostringstream oss{};
    oss << OUTPUT_FILE_PREFIX << timestamp_buffer << get_suffix(m_mode);
    return std::move(oss).str();
}

std::string NestedFileStructure::get_path(const HtsData& hts_data) {
    const auto directory = format_directory(hts_data);
    const auto path = directory / format_filename(hts_data);
    create_output_folder(path);
    return path.string();
};

std::filesystem::path NestedFileStructure::format_directory(const HtsData& hts_data) {
    // clang-format off
    return get_core(hts_data.read_attrs) / 
        format_status(hts_data.read_attrs) /
        format_classification(hts_data.barcoding_result);
    // clang-format on
};

const std::filesystem::path& NestedFileStructure::get_core(const HtsData::ReadAttributes& attrs) {
    // The "core" part of a filepath is common to many reads in a run - cache it to avoid
    // reformatting timestamps etc
    auto& path = m_core_cache[attrs];
    if (!path.empty()) {
        return path;
    }
    path = m_output_dir / format_protocol(attrs) / format_sample(attrs) / format_run(attrs);
    return path;
}

std::string NestedFileStructure::format_protocol(const HtsData::ReadAttributes& attrs) const {
    // The MinKnow spec uses `protocol_group_id` but this is a deprecated field from
    // the FAST5 spec which was changed to `experiment_name`.
    return attrs.experiment_id;
}

std::string NestedFileStructure::format_sample(const HtsData::ReadAttributes& attrs) const {
    return attrs.sample_id;
}

std::string NestedFileStructure::format_run(const HtsData::ReadAttributes& attrs) const {
    const auto datetime = utils::get_minknow_timestamp_from_unix_time(attrs.protocol_start_time_ms);
    // clang-format off
    std::ostringstream oss;
    oss << datetime << "_"
        << attrs.position_id << "_"
        << attrs.flowcell_id << "_"
        << truncate(attrs.protocol_run_id);
    // clang-format on
    return std::move(oss).str();
};

std::string NestedFileStructure::format_status(const HtsData::ReadAttributes& attrs) const {
    std::ostringstream oss;
    oss << status_filetype() << "_" << pass_fail(attrs);
    return std::move(oss).str();
}

std::string NestedFileStructure::format_classification(
        const std::shared_ptr<const BarcodeScoreResult>& result) const {
    if (!result) {
        return "";
    }

    if (result->barcode_name.empty()) {
        return "";
    }

    if (result->barcode_name == UNCLASSIFIED_STR) {
        return UNCLASSIFIED_STR;
    }

    return dorado::barcode_kits::normalize_barcode_name(result->barcode_name);
}

std::string NestedFileStructure::format_filename(const HtsData& hts_data) const {
    std::ostringstream oss;
    // https://nanoporetech.github.io/ont-output-specifications/latest/read_formats/bam/
    // {flow_cell_id}_{status}_{alias_}{short_protocol_run_id}_{short_run_id}_{batch_number}.{filetype}
    // clang-format off
oss << hts_data.read_attrs.flowcell_id << "_" 
    << format_status(hts_data.read_attrs) << "_"
    << alias(hts_data) 
    << truncate(hts_data.read_attrs.protocol_run_id) << "_"
    << truncate(hts_data.read_attrs.acquisition_id) << "_"
    << batch_number()
    << get_suffix(m_mode);
    // clang-format on
    return std::move(oss).str();
};

std::string NestedFileStructure::pass_fail(const HtsData::ReadAttributes& attrs) const {
    return attrs.is_status_pass ? "pass" : "fail";
};

std::string NestedFileStructure::status_filetype() const {
    switch (m_mode) {
    case utils::HtsFile::OutputMode::UBAM:
    case utils::HtsFile::OutputMode::BAM:
    case utils::HtsFile::OutputMode::SAM:
        return "bam";
    case utils::HtsFile::OutputMode::FASTQ:
        return "fastq";
    case utils::HtsFile::OutputMode::FASTA:
        break;
    }

    std::ostringstream oss;
    oss << "NestedFileStructure does not support output format: '" << to_string(m_mode) << "'";
    throw std::logic_error(oss.str());
};

std::string NestedFileStructure::alias(const HtsData& hts_data) const {
    if (!m_sample_sheet) {
        return "";
    }

    if (!hts_data.barcoding_result || hts_data.barcoding_result->barcode_name.empty() ||
        hts_data.barcoding_result->barcode_name == UNCLASSIFIED_STR) {
        return "";
    }

    const auto& attrs = hts_data.read_attrs;
    const auto bc_alias = m_sample_sheet->get_alias(
            attrs.flowcell_id, attrs.position_id, attrs.experiment_id,
            dorado::barcode_kits::normalize_barcode_name(hts_data.barcoding_result->barcode_name));
    return bc_alias + (bc_alias.empty() ? "" : "_");
};

std::string NestedFileStructure::batch_number() const {
    // TODO: Is batch number constant in Dorado?
    return "0";
};

std::string_view NestedFileStructure::truncate(std::string_view field) const {
    return field.substr(0, std::min(field.size(), size_t(8)));
};

}  // namespace hts_writer
}  // namespace dorado