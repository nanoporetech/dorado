#include "remora_utils.h"

namespace dorado {

const std::vector<int> RemoraUtils::BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

}  // namespace dorado
