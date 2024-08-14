#pragma once

namespace dorado::utils {

struct Overlap {
    int qstart{0};
    int qend{0};
    int qlen{0};
    int tstart{0};
    int tend{0};
    int tlen{0};
    bool fwd{false};
};

}  // namespace dorado::utils