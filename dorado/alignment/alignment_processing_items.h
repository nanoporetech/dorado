#pragma once

#include "utils/hts_file.h"

#include <string>
#include <vector>

namespace dorado::alignment {

struct AlignmentProcessingInfo {
    AlignmentProcessingInfo() {}
    AlignmentProcessingInfo(std::string input_,
                            std::string output_,
                            utils::HtsFile::OutputMode output_mode_)
            : input(std::move(input_)), output(std::move(output_)), output_mode(output_mode_) {}
    std::string input{};
    std::string output{};
    utils::HtsFile::OutputMode output_mode{};
};

class AlignmentProcessingItems {
    const std::string m_input_path;
    const std::string m_output_folder;
    bool m_recursive_input;
    bool m_allow_output_to_folder_from_stdin;

    std::vector<AlignmentProcessingInfo> m_processing_list{};

    void add_all_valid_files();

    bool check_recursive_arg_false();

    bool try_create_output_folder();

    bool check_output_folder_for_input_folder(const std::string& input_folder);

    bool initialise_for_stdin();

    bool initialise_for_file();

    bool initialise_for_folder();

public:
    AlignmentProcessingItems(std::string input_path,
                             bool recursive_input,
                             std::string output_folder,
                             bool allow_output_to_folder_from_stdin);

    bool initialise();

    const std::vector<AlignmentProcessingInfo>& get() const { return m_processing_list; }
};

}  // namespace dorado::alignment