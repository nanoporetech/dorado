#pragma once
#include "read_pipeline/ReadPipeline.h"
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
    AdapterDetector();
    ~AdapterDetector();

    AdapterScoreResult find_adapters(const std::string& seq);
    AdapterScoreResult find_primers(const std::string& seq);

    struct Query {
        std::string name;
        std::string sequence;
        std::string sequence_rev;
    };

    const std::vector<Query>& get_adapter_sequences() const;
    const std::vector<Query>& get_primer_sequences() const;

    static void check_and_update_barcoding(SimplexRead& read, std::pair<int, int>& trim_interval);

private:
    enum QueryType { ADAPTER, PRIMER };

    std::vector<Query> m_adapter_sequences;
    std::vector<Query> m_primer_sequences;
    AdapterScoreResult detect(const std::string& seq,
                              const std::vector<Query>& queries,
                              QueryType query_type) const;
};

}  // namespace demux

}  // namespace dorado
