#include "utils/alignment_utils.h"

#include <minimap.h>

namespace dorado::utils {

std::string alignment_to_str(const char* query,
                             const char* target,
                             const EdlibAlignResult& result) {
    std::string ss;
    ss.reserve(result.alignmentLength * 3 + 2);

    int qpos = 0;
    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss += "-";
        } else {
            ss += query[qpos];
            qpos++;
        }
    }

    ss += '\n';

    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_MATCH) {
            ss += "|";
        } else if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss += " ";
        } else if (result.alignment[i] == EDLIB_EDOP_DELETE) {
            ss += " ";
        } else if (result.alignment[i] == EDLIB_EDOP_MISMATCH) {
            ss += "*";
        }
    }

    ss += '\n';

    int tpos = result.startLocations[0];
    for (int i = 0; i < result.alignmentLength; i++) {
        if (result.alignment[i] == EDLIB_EDOP_INSERT) {
            ss += "-";
        } else {
            ss += target[tpos];
            tpos++;
        }
    }

    return ss;
}

}  // namespace dorado::utils
