#pragma once
#include "ReadSplitter.h"
#include "splitter/splitter_utils.h"
#include "utils/types.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace dorado::splitter {

class DuplexReadSplitter : public ReadSplitter {
public:
    DuplexReadSplitter(DuplexSplitSettings settings);
    ~DuplexReadSplitter();

    std::vector<SimplexReadPtr> split(SimplexReadPtr init_read) const override;

private:
    struct ExtRead;
    using SplitFinderF = std::function<splitter::PosRanges(const ExtRead&)>;

    ExtRead create_ext_read(SimplexReadPtr r) const;
    PosRanges possible_pore_regions(const ExtRead& read) const;
    PosRanges find_muA_adapter_spikes(const ExtRead& read) const;
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
