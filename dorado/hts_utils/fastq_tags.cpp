#include "hts_utils/fastq_tags.h"

#include "utils/string_utils.h"

#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

namespace dorado::utils {

ReadGroupData parse_rg_from_hts_tags(const std::string_view tag_str) {
    /// Example:
    ///     @0a579f20-1ab0-4e5b-91e6-30e37cf7eb96   RG:Z:4524e8b9-b90e-4ffb-a13a-380266513b64_dna_r10.4.1_e8.2_400bps_hac@v5.0.0    ch:i:63 st:Z:2022-10-18T10:38:07.247+00:00      PU:Z:PAM93185   LB:Z:PCR_zymo   SM:Z:barcode03  al:Z:alias_for_bc03     pi:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5 DS:Z:gpu:Tesla V100-PCIE-16GB   qs:f:30.0       mx:i:2  rn:i:432        ts:i:1048       pt:i:120        pa:B:i,12,30,45,232,242
    /// IMPORTANT:
    ///     - Modbase models are not supported in the FASTQ output.

    constexpr std::string_view KEY_READ_GROUP{"RG:Z:"};
    constexpr std::string_view KEY_FLOWCELL_ID{"PU:Z:"};
    constexpr std::string_view KEY_DEVICE_ID{"PM:Z:"};
    constexpr std::string_view KEY_EXPERIMENT_START_TIME{"st:Z:"};
    constexpr std::string_view KEY_SAMPLE_ID{"LB:Z:"};

    const std::regex pattern(R"(^([0-9a-f\-]{1,})_(.*@v\d+\.\d+\.\d+)(.*)$)");

    const std::unordered_set<std::string_view> key_set{
            KEY_READ_GROUP, KEY_FLOWCELL_ID, KEY_DEVICE_ID, KEY_EXPERIMENT_START_TIME,
            KEY_SAMPLE_ID,
    };

    const std::vector<std::string_view> tokens = dorado::utils::split_view(tag_str, '\t');

    ReadGroupData ret;

    for (const std::string_view token : tokens) {
        // Filter tags which do not match our keys of interest.
        if (std::size(token) < 5) {
            continue;
        }
        if (key_set.count(token.substr(0, 5)) == 0) {
            continue;
        }

        const std::string_view val = token.substr(5);

        if (token.starts_with(KEY_READ_GROUP)) {
            // Examples:
            //  RG:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5_rna004_130bps_fast@v5.1.0_barcode02
            //  RG:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5_unknown_barcode02
            //  RG:Z:e4994c62-93f9-439a-bc8f-d20c95a137a5_rna004_130bps_fast@v5.1.0_29d8704b
            ret.id = val;
            std::smatch match;
            const std::string val_str{val};
            if (std::regex_match(val_str, match, pattern)) {
                // NOTE: match[3] is an extra suffix, such as the read group ID.
                ret.data.run_id = match[1];
                ret.data.basecalling_model = match[2];
                ret.found = true;
            }
        } else if (token.starts_with(KEY_FLOWCELL_ID)) {
            // Example: PU:Z:PAM93185
            ret.data.flowcell_id = val;
            ret.found = true;

        } else if (token.starts_with(KEY_DEVICE_ID)) {
            // Example: PM:Z:MN12345
            ret.data.device_id = val;
            ret.found = true;

        } else if (token.starts_with(KEY_EXPERIMENT_START_TIME)) {
            // Example: st:Z:2022-10-18T10:38:07.247+00:00
            ret.data.exp_start_time = val;
            ret.found = true;

        } else if (token.starts_with(KEY_SAMPLE_ID)) {
            // Example: LB:Z:PCR_zymo
            ret.data.sample_id = val;
            ret.found = true;
        }
    }

    return ret;
}

}  // namespace dorado::utils