#pragma once
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

    AdapterScoreResult find_adapters(const std::string& seq) const;
    AdapterScoreResult find_primers(const std::string& seq) const;

    struct Query {
        std::string name;
        std::string sequence;
        std::string sequence_rev;
        bool operator<(const Query& rhs) const { return name < rhs.name; }
    };

    const std::vector<Query>& get_adapter_sequences() const;
    const std::vector<Query>& get_primer_sequences() const;

private:
    enum QueryType { ADAPTER, PRIMER };

    std::vector<Query> m_adapter_sequences;
    std::vector<Query> m_primer_sequences;
    AdapterScoreResult detect(const std::string& seq,
                              const std::vector<Query>& queries,
                              QueryType query_type) const;
    void parse_custom_sequence_file(const std::string& custom_sequence_file);
};

}  // namespace demux
}  // namespace dorado
