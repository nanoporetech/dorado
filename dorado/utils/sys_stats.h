#pragma once

#include "stats.h"

#include <string>
#include <tuple>

namespace dorado {
namespace stats {

std::tuple<std::string, NamedStats> sys_stats_report();

}  // namespace stats
}  // namespace dorado