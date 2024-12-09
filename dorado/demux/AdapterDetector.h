#pragma once
#include "utils/adapter_primer_kits.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <mutex>
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

    using Query = dorado::adapter_primer_kits::Candidate;

    std::vector<Query>& get_adapter_sequences(const std::string& kit_name);
    std::vector<Query>& get_primer_sequences(const std::string& kit_name);

private:
    enum QueryType { ADAPTER, PRIMER };

    std::mutex m_mutex;
    std::unique_ptr<dorado::adapter_primer_kits::AdapterPrimerManager> m_sequence_manager;
    std::unordered_map<std::string, std::vector<Query>> m_adapter_sequences;
    std::unordered_map<std::string, std::vector<Query>> m_primer_sequences;
    AdapterScoreResult detect(const std::string& seq,
                              const std::vector<Query>& queries,
                              QueryType query_type) const;
};

}  // namespace demux
}  // namespace dorado
