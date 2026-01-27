#pragma once

#include <cstdint>

namespace dorado::secondary {

struct KadayashiOptions {
    bool disable_interval_expansion{false};
    int32_t min_base_quality{5};
    int32_t min_varcall_coverage{5};
    float min_varcall_fraction{0.2f};
    int32_t max_clipping{200};
    int32_t min_strand_cov{3};
    float min_strand_cov_frac{0.03f};
    float max_gapcompressed_seqdiv{0.1f};
    bool use_dvr_for_phasing{false};
};

}  // namespace dorado::secondary
