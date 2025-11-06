#pragma once

#include "hts_utils/hts_types.h"
#include "hts_writer/HtsFileWriter.h"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace dorado::hts_writer {

class IStructure {
public:
    virtual ~IStructure() = default;
    virtual std::string get_path(const HtsData &item) = 0;
};

class SingleFileStructure : public IStructure {
public:
    SingleFileStructure(const std::string &output_dir, OutputMode mode);
    std::string get_path([[maybe_unused]] const HtsData &hts_data) override;

private:
    const OutputMode m_mode;
    const std::filesystem::path m_path;

    std::string get_filename() const;
};

class NestedFileStructure : public IStructure {
public:
    NestedFileStructure(const std::string &output_dir, OutputMode mode, bool assume_barcodes)
            : m_output_dir(std::filesystem::weakly_canonical(std::filesystem::path(output_dir))),
              m_mode(mode),
              m_assume_barcodes(assume_barcodes) {};

    std::string get_path(const HtsData &hts_data) override;

private:
    const std::filesystem::path m_output_dir;
    const OutputMode m_mode;

    const bool m_assume_barcodes;

    std::unordered_map<HtsData::ReadAttributes,
                       std::filesystem::path,
                       HtsData::ReadAttributesCoreHasher,
                       HtsData::ReadAttributesCoreComparator>
            m_core_cache;

    /*
    https://nanoporetech.github.io/ont-output-specifications/latest/minknow/output_structure/
    
    The "core" part of the filepath is common to many reads in a run - cache this part.
    C | <root>            output_folder/
    O | <protocol>        -- {protocol_group_id} (if it exists is source pod5 files)/
    R | <sample>          | --{sample_id}/
    E | <run>             | | --{start_time}_{device_id}_{flow_cell_id}_{short_protocol_run_id}/
      | <filetype status> | | | --{filetype}{_status}/
      | <alias>           | | | | --{alias} (if barcoding, otherwise folder is absent)
    */

    // Format all subdirectories
    std::filesystem::path format_directory(const HtsData &hts_data, const std::string &alias);
    /*
    https://nanoporetech.github.io/ont-output-specifications/latest/read_formats/bam/
    `{flow_cell_id}{status}_{alias_}{short_protocol_run_id}_{short_run_id}_{batch_number}.{filetype}`
    */
    std::string format_filename(const HtsData &hts_data, const std::string &alias) const;

    // Format and cache "core" subdirectory `{root}/{protocol_group_id}/{sample}/{run}/`
    const std::filesystem::path &get_core(const HtsData::ReadAttributes &attrs);
    // Format the <protocol> subdirectory `{protocol_group_id}/` - this is now an alias for `experiment_name`
    std::string format_protocol(const HtsData::ReadAttributes &attrs) const;
    // Format the <sample> subdirectory `{sample_id}/`
    std::string format_sample(const HtsData::ReadAttributes &attrs) const;
    // Format the <run> subdirectory - `{start_time}_{device_id}_{flow_cell_id}_{short_protocol_run_id}/`
    std::string format_run(const HtsData::ReadAttributes &attrs) const;
    // Format the <filetype status> subdirectory `{filetype}{_status}/`
    std::string format_filetype_status(const HtsData::ReadAttributes &attrs) const;
    /* Format the barcode alias
        Returns - Empty string if no barcoding done
        Returns - Alias if set, else barcode name otherwise including `unclassified`
    */
    std::string format_alias(const HtsData &hts_data) const;

    // Get the filetype prefix used in the status
    std::string filetype() const;
    // Get the pass/fail status of this read with leading underscore
    std::string status(const HtsData::ReadAttributes &attrs) const;
    // Get the batch numer
    std::string batch_number() const;
    // Truncate a field to at most 8 characters
    std::string_view truncate(std::string_view field) const;
};

}  // namespace dorado::hts_writer
