#include "parse_custom_sequences.h"

#include "utils/fasta_reader.h"

#include <stdexcept>
#include <tuple>

namespace {
std::pair<std::string, std::string> parse_token(const std::string& token) {
    auto split_pos = token.find('=');
    if (split_pos == 0 || split_pos == token.size() - 1 || split_pos == std::string::npos) {
        return {};
    }
    auto key = token.substr(0, split_pos);
    auto value = token.substr(split_pos + 1);
    return {key, value};
}
}  // namespace

namespace dorado::demux {

std::vector<CustomSequence> parse_custom_sequences(const std::string& sequences_file) {
    utils::FastaReader reader(sequences_file);
    if (!reader.is_valid()) {
        throw std::runtime_error("Failed to extract sequences from '" + sequences_file + "'.");
    }
    std::vector<CustomSequence> sequences;
    while (true) {
        auto record = reader.try_get_next_record();
        if (!record) {
            break;
        }
        CustomSequence custom;
        custom.name = record->record_name();
        custom.sequence = record->sequence();
        const auto& tokens = record->get_tokens();
        if (tokens.size() > 1) {
            for (auto iter = tokens.begin() + 1; iter != tokens.end(); ++iter) {
                auto [key, value] = parse_token(*iter);
                if (key.empty()) {
                    custom.tags.clear();
                    break;
                }
                custom.tags.emplace(std::move(key), std::move(value));
            }
        }
        sequences.push_back(std::move(custom));
    }
    return sequences;
}

}  // namespace dorado::demux
