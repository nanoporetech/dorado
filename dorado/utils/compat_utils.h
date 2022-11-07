#pragma once

#ifdef _WIN32

namespace dorado::utils {
char* strptime(const char* s, const char* f, struct tm* tm);
}

using dorado::utils::strptime;

#endif  // _WIN32
