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

class DuplexSplitNode : public MessageSink {
public:
    typedef std::pair<uint64_t, uint64_t> PosRange;
    typedef std::vector<PosRange> PosRanges;

    DuplexSplitNode(DuplexSplitSettings settings,
                    int num_worker_threads = 5,
                    size_t max_reads = 1000);
    ~DuplexSplitNode() { terminate_impl(); }
    std::string get_name() const override { return "DuplexSplitNode"; }
    stats::NamedStats sample_stats() const override;
    void terminate(const FlushOptions& flush_options) override { terminate_impl(); }
    void restart() override;

    std::vector<std::shared_ptr<Read>> split(std::shared_ptr<Read> init_read) const;

private:
    void start_threads();
    void terminate_impl();
    //TODO consider precomputing and reusing ranges with high signal
    struct ExtRead {
        std::shared_ptr<Read> read;
        torch::Tensor data_as_float32;
        std::vector<uint64_t> move_sums;
        PosRanges possible_pore_regions;
    };

    typedef std::function<PosRanges(const ExtRead&)> SplitFinderF;

    ExtRead create_ext_read(std::shared_ptr<Read> r) const;
    std::vector<PosRange> possible_pore_regions(const ExtRead& read) const;
    bool check_nearby_adapter(const Read& read, PosRange r, int adapter_edist) const;
    std::optional<std::pair<PosRange, PosRange>> check_flank_match(const Read& read,
                                                                   PosRange r,
                                                                   float err_thr) const;
    std::optional<PosRange> identify_middle_adapter_split(const Read& read) const;
    std::optional<PosRange> identify_extra_middle_split(const Read& read) const;

    std::vector<std::shared_ptr<Read>> subreads(std::shared_ptr<Read> read,
                                                const PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>> build_split_finders() const;

    void worker_thread();  // Worker thread performs splitting asynchronously.

    const DuplexSplitSettings m_settings;
    std::vector<std::pair<std::string, SplitFinderF>> m_split_finders;
    std::atomic<size_t> m_active{0};
    const int m_num_worker_threads;
    std::vector<std::unique_ptr<std::thread>> m_worker_threads;
};

}  // namespace dorado
