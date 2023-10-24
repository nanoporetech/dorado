#pragma once
#include "ReadPipeline.h"
#include "utils/stats.h"
#include "utils/types.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dorado {

struct RNASplitSettings {
    int16_t pore_thr = 1500;
    uint64_t pore_cl_dist = 4000;  // TODO maybe use frequency * 1sec here?
    //maximal 'open pore' region to consider (bp)
    uint64_t max_pore_region = 500;
    //in samples
    uint64_t expect_pore_prefix = 2000;
};

class RNASplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;
    typedef std::vector<PosRange> PosRanges;

    RNASplitNode(RNASplitSettings settings, int num_worker_threads = 5, size_t max_reads = 1000);
    ~RNASplitNode() { terminate_impl(); }
    std::string get_name() const override { return "RNASplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    std::vector<SimplexReadPtr> split(SimplexReadPtr init_read) const;

private:
    void start_threads();
    void terminate_impl();
    //TODO consider precomputing and reusing ranges with high signal
    struct ExtRead {
        SimplexReadPtr read;
        torch::Tensor data_as_int16;
        PosRanges possible_pore_regions;
    };

    typedef std::function<PosRanges(const ExtRead&)> SplitFinderF;

    ExtRead create_ext_read(SimplexReadPtr r) const;
    std::vector<PosRange> possible_pore_regions(const ExtRead& read) const;

    std::vector<SimplexReadPtr> subreads(SimplexReadPtr read, const PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>> build_split_finders() const;

    void worker_thread();  // Worker thread performs splitting asynchronously.

    const RNASplitSettings m_settings;
    std::vector<std::pair<std::string, SplitFinderF>> m_split_finders;
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
};

}  // namespace dorado
