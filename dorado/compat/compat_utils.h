#pragma once

#ifdef _WIN32
#include <cstdlib>

struct tm;

namespace dorado::utils {
char* strptime(const char* s, const char* f, tm* tm);

// A simple wrapper for setenv, since windows doesn't have it.
int setenv(const char* name, const char* value, int overwrite);

}  // namespace dorado::utils

using dorado::utils::setenv;
using dorado::utils::strptime;

#endif  // _WIN32

namespace dorado::utils {
// Ensures std::setlocale(LC_ALL, "") will succeed.
// See DOR-234 for details.
void ensure_user_locale_may_be_set_DOR_234();
}  // namespace dorado::utils