#include "bam_info.h"

#include "bam_file.h"
#include "utils/container_utils.h"
#include "utils/string_utils.h"

#include <htslib/sam.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace dorado::secondary {

BamInfo analyze_bam(const std::filesystem::path& in_aln_bam_fn, const std::string& cli_read_group) {
    BamInfo ret;

    BamFile bam(in_aln_bam_fn);

    const std::vector<HeaderLineData> header = bam.parse_header();

    // Get info from headers: program and the read groups.
    for (const auto& line : header) {
        // Convert all tags into a lookup.
        const std::unordered_map<std::string, std::string> tags = [&]() {
            std::unordered_map<std::string, std::string> local_ret;
            for (const auto& [key, value] : line.tags) {
                local_ret[key] = value;
            }
            return local_ret;
        }();

        if (line.header_type == "@PG") {
            // Example PG line:
            //      @PG	ID:aligner	PP:samtools.2	PN:dorado	VN:0.0.0+2852e11d	DS:2.27-r1193

            const auto& it_pn = tags.find("PN");
            const auto& it_id = tags.find("ID");
            if ((it_pn != std::end(tags)) && it_id != std::end(tags)) {
                // Convert the program name to lowercase just in case.
                std::string pn = it_pn->second;
                std::transform(std::begin(pn), std::end(pn), std::begin(pn),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                // Convert the program ID to lowercase just in case.
                std::string id = it_id->second;
                std::transform(std::begin(id), std::end(id), std::begin(id),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                if ((pn == "dorado") && utils::starts_with(id, "aligner")) {
                    // Multiple tools can be run on a BAM, and the ID field needs to be unique by spec.
                    // Example possibilites: aligner, aligner.1, samtools.1, samtools.2, etc.
                    ret.uses_dorado_aligner = true;
                }
            }
        } else if (line.header_type == "@RG") {
            // Example RG line:
            //      @RG	ID:e705d8cfbbe8a6bc43a865c71ace09553e8f15cd_dna_r10.4.1_e8.2_400bps_hac@v5.0.0	DT:2022-10-18T10:38:07.247961+00:00	DS:runid=e705d8cfbbe8a6bc43a865c71ace09553e8f15cd basecall_model=dna_r10.4.1_e8.2_400bps_hac@v5.0.0 modbase_models=dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mC_5hmC@v2,dna_r10.4.1_e8.2_400bps_hac@v5.0.0_6mA@v2	LB:PCR_zymo	PL:ONT	PM:4A	PU:PAM93185	al:PCR_zymo

            // Parse the read group ID.
            const auto& it_id = tags.find("ID");
            const std::string id = (it_id != std::end(tags)) ? it_id->second : "";

            // Parse the basecaller model.
            const auto& it_ds = tags.find("DS");
            std::string basecaller_model;
            if (it_ds != std::end(tags)) {
                const std::vector<std::string> tokens = utils::split(it_ds->second, ' ');
                constexpr std::string_view TOKEN_NAME{"basecall_model="};
                for (const auto& token : tokens) {
                    if (!utils::starts_with(token, TOKEN_NAME)) {
                        continue;
                    }
                    basecaller_model = token.substr(std::size(TOKEN_NAME));
                    break;
                }
            }

            if (std::empty(id)) {
                continue;
            }
            if (!std::empty(cli_read_group) && (id != cli_read_group)) {
                continue;
            }
            if (std::empty(basecaller_model)) {
                continue;
            }

            ret.read_groups.emplace(id);
            ret.basecaller_models.emplace(basecaller_model);
        }
    }

    // Check for the dwells ("mv") tag. Only parse one record.
    {
        const auto record = bam.get_next();
        if ((record != nullptr) && (bam_aux_get(record.get(), "mv") != nullptr)) {
            ret.has_dwells = true;
        }
    }

    return ret;
}

void check_read_groups(const BamInfo& bam_info, const std::string& cli_read_group) {
    if (!std::empty(cli_read_group) && std::empty(bam_info.read_groups)) {
        throw std::runtime_error{
                "No @RG headers found in the input BAM, but user-specified RG was given. RG: '" +
                cli_read_group + "'"};

    } else if (std::empty(cli_read_group) && std::size(bam_info.read_groups) > 1) {
        throw std::runtime_error{
                "The input BAM contains more than one read group. Please specify --RG to select "
                "which read group to process."};

    } else if (!std::empty(cli_read_group) && !std::empty(bam_info.read_groups)) {
        if (bam_info.read_groups.count(cli_read_group) == 0) {
            std::ostringstream oss;
            utils::print_container(oss, bam_info.read_groups, ", ");
            throw std::runtime_error{"Requested RG is not in the input BAM. Requested: '" +
                                     cli_read_group + "'"};
        }
    }
}

}  // namespace dorado::secondary
