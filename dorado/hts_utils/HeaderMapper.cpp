#include "hts_utils/HeaderMapper.h"

#include "hts_utils/fastq_reader.h"
#include "hts_utils/hts_types.h"

#include <htslib/sam.h>

#include <filesystem>

namespace {}  // anonymous namespace

namespace dorado::utils {

HeaderMapper::HeaderMapper(const std::vector<std::filesystem::path>& inputs, bool strip_alignment)
        : m_strip_alignment(strip_alignment) {
    process(inputs);
}

void HeaderMapper::process(const std::vector<std::filesystem::path>& inputs) {
    // HtsData::ReadAttributes {
    //     std::string sequencing_kit{};
    //     std::string experiment_id{};
    //     std::string sample_id{};
    //     std::string position_id{};
    //     std::string flowcell_id{};
    //     std::string protocol_run_id{};
    //     std::string acquisition_id{};
    //     int64_t protocol_start_time_ms{0};
    //     std::size_t subread_id{0};
    //     bool is_status_pass{true};
    // };
    for (const auto& input : inputs) {
        if (is_fastq(input)) {
            // proceess FASTQ
        } else {
            // must be htslib
        }
    }
};

}  // namespace dorado::utils
