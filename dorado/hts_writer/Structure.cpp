#include "hts_writer/Structure.h"

#include "hts_utils/hts_file.h"
#include "hts_utils/hts_types.h"
#include "utils/SampleSheet.h"
#include "utils/barcode_kits.h"
#include "utils/time_utils.h"

#include <htslib/sam.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

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
        });
    }
#endif

    if (std::filesystem::exists(path.parent_path())) {
        return;
    }

    spdlog::debug("Creating output folder: '{}'. Length:{}", path.parent_path().string(),
                  path.string().size());
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
    const auto alias = format_alias(hts_data);
    const auto directory = format_directory(hts_data, alias);
    const auto path = directory / format_filename(hts_data, alias);
    create_output_folder(path);
    return path.string();
}

std::filesystem::path NestedFileStructure::format_directory(const HtsData& hts_data,
                                                            const std::string& alias) {
    return get_core(hts_data.read_attrs) / format_filetype_status(hts_data.read_attrs) / alias;
}

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
    const auto datetime =
            utils::get_minknow_timestamp_from_unix_time_ms(attrs.protocol_start_time_ms);
    std::ostringstream oss;
    oss << datetime << "_";
    oss << attrs.position_id << "_";
    oss << attrs.flowcell_id << "_";
    oss << truncate(attrs.protocol_run_id);
    return std::move(oss).str();
}

std::string NestedFileStructure::format_filetype_status(
        const HtsData::ReadAttributes& attrs) const {
    std::ostringstream oss;
    oss << filetype() << status(attrs);
    return std::move(oss).str();
}

std::string NestedFileStructure::format_filename(const HtsData& hts_data,
                                                 const std::string& alias) const {
    // https://nanoporetech.github.io/ont-output-specifications/latest/read_formats/bam/
    // {flow_cell_id}{_status}_{alias_}{short_protocol_run_id}_{short_run_id}_{batch_number}.{filetype}
    // const auto alias = format_alias(hts_data);
    std::ostringstream oss;
    oss << hts_data.read_attrs.flowcell_id;
    oss << status(hts_data.read_attrs) << "_";
    oss << alias << (!alias.empty() ? "_" : "");
    oss << truncate(hts_data.read_attrs.protocol_run_id) << "_";
    oss << truncate(hts_data.read_attrs.acquisition_id) << "_";
    oss << batch_number();
    oss << get_suffix(m_mode);
    return std::move(oss).str();
};

std::string NestedFileStructure::status(const HtsData::ReadAttributes& attrs) const {
    return attrs.is_status_pass ? "_pass" : "_fail";
};

std::string NestedFileStructure::filetype() const {
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

std::string NestedFileStructure::format_alias(const HtsData& hts_data) const {
    std::string_view barcode_name;
    if (hts_data.barcoding_result && !hts_data.barcoding_result->barcode_name.empty()) {
        barcode_name = hts_data.barcoding_result->barcode_name;
    } else if (hts_data.bam_ptr) {
        // No barcoding result - check the BC tag in case this is a barcoded read we've read in from file
        const auto bc_tag = bam_aux_get(hts_data.bam_ptr.get(), "BC");
        if (bc_tag) {
            barcode_name = bam_aux2Z(bc_tag);
            for (const auto& [kit_name, kit_info] : barcode_kits::get_kit_infos()) {
                if (barcode_name.starts_with(kit_name) &&
                    barcode_name.size() > kit_name.size() + 1) {
                    // strip the kit name from the barcode
                    barcode_name = barcode_name.substr(kit_name.size() + 1);
                    break;
                }
            }
            return std::string(barcode_name);
        }
        // Unclassified reads read from file won't have a BC tag. If we've been told this is a demux operation,
        // ensure these reads are correctly placed in the unclassified folder
        if (m_assume_barcodes) {
            return UNCLASSIFIED_STR;
        }
    }

    // No barcode
    if (barcode_name.empty()) {
        return {};
    }

    // Always return "unclassified" as there's no alias
    if (barcode_name == UNCLASSIFIED_STR) {
        return UNCLASSIFIED_STR;
    }

    // Return the alias if found otherwise fall back to the barcode name
    const auto norm_barcode_name = barcode_kits::normalize_barcode_name(barcode_name);
    if (m_sample_sheet) {
        const auto& attrs = hts_data.read_attrs;
        const auto bc_alias = m_sample_sheet->get_alias(attrs.flowcell_id, attrs.position_id,
                                                        attrs.experiment_id, norm_barcode_name);
        return !bc_alias.empty() ? bc_alias : norm_barcode_name;
    }

    return norm_barcode_name;
};

std::string NestedFileStructure::batch_number() const { return "0"; };

std::string_view NestedFileStructure::truncate(std::string_view field) const {
    return field.substr(0, std::min(field.size(), size_t(8)));
};

}  // namespace hts_writer
}  // namespace dorado