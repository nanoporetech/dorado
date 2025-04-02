#include "sys_utils.h"

#include <fstream>
#include <string>

namespace dorado::utils {

bool running_in_docker() {
#if defined(__linux__)
    // Look for docker paths in the init process.
    std::ifstream cgroup_file("/proc/1/cgroup", std::ios_base::in | std::ios_base::binary);
    if (cgroup_file.is_open()) {
        std::string line;
        while (std::getline(cgroup_file, line)) {
            if (line.find(":/docker/") != line.npos) {
                return true;
            }
        }
    }
#endif
    return false;
}

}  // namespace dorado::utils
