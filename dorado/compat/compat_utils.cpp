#include "compat_utils.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <ctime>
#include <iomanip>
#include <sstream>

namespace dorado::utils {
char* strptime(const char* s, const char* f, tm* tm) {
    std::istringstream input(s);
    input.imbue(std::locale(setlocale(LC_ALL, nullptr)));
    input >> std::get_time(tm, f);
    if (input.fail()) {
        return nullptr;
    }
    return (char*)(s + input.tellg());
}

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

void ensure_user_locale_may_be_set_DOR_234() {
    // JIRA issue https://nanoporetech.atlassian.net/browse/DOR-234
    // The indicators library tries to set the user preferred locale, so we ensure
    // this is possible.
    // An invalid value for the LANG environment variable (e.g. if it's not present
    // on the machine) may cause setting the locale the the user preference with
    // setlocale(LC_ALL, "") to fail and return null.
    // So test for null and provide a valid LANG value if needed
    // For fallback behaviour of setlocale(LC_ALL, "") see:
    // https://man7.org/linux/man-pages/man3/setlocale.3.html

    // Passing nullptr as the locale will cause the current default value to be returned
    // for a c++ program this should have been set to "C" during program startup
    auto default_locale_to_restore = std::setlocale(LC_NUMERIC, nullptr);

    // passing "" as the locale will set and return the user preferred locale (if valid)
    auto user_preferred_locale = std::setlocale(LC_ALL, "");
    if (!user_preferred_locale) {
        // The user preferred locale could not be set.
        // Provide a valid value for LANG so that it may be correctly set with ""
        setenv("LANG", "C", true);
    } else {
        // restore the original default locale
        std::setlocale(LC_ALL, default_locale_to_restore);
    }
}

}  // namespace dorado::utils
#endif  // _WIN32
