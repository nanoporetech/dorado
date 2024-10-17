#pragma once
#include "read_pipeline/messages.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace dorado {

namespace demux {

class AdapterDetector {
public:
    AdapterDetector(const std::optional<std::string>& custom_primer_file);
    ~AdapterDetector();

    AdapterScoreResult find_adapters(const std::string& seq, const std::string& kit_name);
    AdapterScoreResult find_primers(const std::string& seq, const std::string& kit_name);

    struct Query {
        std::string name;
        std::string sequence;
        std::string sequence_rev;
        bool operator<(const Query& rhs) const { return name < rhs.name; }
    };

    std::vector<Query>& get_adapter_sequences(const std::string& kit_name);
    std::vector<Query>& get_primer_sequences(const std::string& kit_name);

private:
    enum QueryType { ADAPTER, PRIMER };

    std::mutex m_mutex;
    std::unordered_map<std::string, std::vector<Query>> m_adapter_sequences;
    std::unordered_map<std::string, std::vector<Query>> m_primer_sequences;
    std::vector<Query> m_custom_primer_sequences;
    AdapterScoreResult detect(const std::string& seq,
                              const std::vector<Query>& queries,
                              QueryType query_type) const;
    void parse_custom_sequence_file(const std::string& custom_sequence_file);
};

}  // namespace demux
}  // namespace dorado
