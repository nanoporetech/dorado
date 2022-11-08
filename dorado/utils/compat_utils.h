#pragma once

#ifdef _WIN32

struct tm;

namespace dorado::utils {
char* strptime(const char* s, const char* f, tm* tm);
}

using dorado::utils::strptime;

#endif  // _WIN32
