#pragma once

#ifdef _WIN32
#include <cstdlib>

struct tm;

namespace dorado::utils {
char* strptime(const char* s, const char* f, tm* tm);
}

using dorado::utils::strptime;

// A simple wrapper for setenv, since windows doesn't have it.
int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite) {
        size_t envsize = 0;
        int errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) {
            return errcode;
        }
    }
    return _putenv_s(name, value);
}

#endif  // _WIN32
