#pragma once

#include "edlib.h"

#include <string>

namespace dorado::utils {

std::string print_alignment(const char* query, const char* target, const EdlibAlignResult& result);

}