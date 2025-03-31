#pragma once
#include "ReadSplitter.h"
#include "splitter_utils.h"
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

    std::vector<ExtRead> apply_split_finders(ExtRead read) const;
    template <typename SplitFinder>
    void apply_split_finder(std::vector<ExtRead>& to_split,
                            [[maybe_unused]] const char* description,
                            const SplitFinder& split_finder) const;

    const DuplexSplitSettings m_settings;
};

}  // namespace dorado::splitter
