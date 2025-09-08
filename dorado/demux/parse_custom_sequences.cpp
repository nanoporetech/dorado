#include "demux/parse_custom_sequences.h"

#include "hts_utils/bam_utils.h"
#include "read_pipeline/base/HtsReader.h"

#include <htslib/sam.h>

namespace dorado::demux {

std::vector<CustomSequence> parse_custom_sequences(const std::string& sequences_file) {
    HtsReader reader(sequences_file, std::nullopt);
    std::vector<CustomSequence> sequences;
    while (reader.read()) {
        CustomSequence custom;
        custom.name = bam_get_qname(reader.record.get());
        custom.sequence = utils::extract_sequence(reader.record.get());
        const auto& tag_list = get_custom_sequence_tag_list();
        for (const auto& tag : tag_list) {
            if (reader.has_tag(tag.c_str())) {
                auto value = reader.get_tag<std::string>(tag.c_str());
                custom.tags.emplace(tag, std::move(value));
            }
        }
        sequences.emplace_back(std::move(custom));
    }
    return sequences;
}

const std::vector<std::string>& get_custom_sequence_tag_list() {
    static std::vector<std::string> tag_list = {
            "et",  // Sequence entry type tag. For use in adapter/primer custom sequence files.
            "sk",  // Sequencing kits tag. For use in adapter/primer custom sequence files.
    };
    return tag_list;
}

}  // namespace dorado::demux
