#pragma once
#include "ReadSplitter.h"
#include "read_pipeline/ReadPipeline.h"
#include "splitter/splitter_utils.h"
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

namespace dorado::splitter {

class DuplexSplitNode : public ReadSplitter {
public:
    DuplexSplitNode(DuplexSplitSettings settings);
    ~DuplexSplitNode() {}

    std::vector<SimplexReadPtr> split(SimplexReadPtr init_read) const override;

private:
    //TODO consider precomputing and reusing ranges with high signal
    struct ExtRead {
        SimplexReadPtr read;
        torch::Tensor data_as_float32;
        std::vector<uint64_t> move_sums;
        splitter::PosRanges possible_pore_regions;
    };

    using SplitFinderF = std::function<splitter::PosRanges(const ExtRead&)>;

    ExtRead create_ext_read(SimplexReadPtr r) const;
    std::vector<splitter::PosRange> possible_pore_regions(const ExtRead& read) const;
    bool check_nearby_adapter(const SimplexRead& read,
                              splitter::PosRange r,
                              int adapter_edist) const;
    std::optional<std::pair<splitter::PosRange, splitter::PosRange>>
    check_flank_match(const SimplexRead& read, splitter::PosRange r, float err_thr) const;
    std::optional<splitter::PosRange> identify_middle_adapter_split(const SimplexRead& read) const;
    std::optional<splitter::PosRange> identify_extra_middle_split(const SimplexRead& read) const;

    std::vector<SimplexReadPtr> subreads(SimplexReadPtr read,
                                         const splitter::PosRanges& spacers) const;

    std::vector<std::pair<std::string, SplitFinderF>> build_split_finders() const;

    const DuplexSplitSettings m_settings;
    std::vector<std::pair<std::string, SplitFinderF>> m_split_finders;
};

}  // namespace dorado::splitter
