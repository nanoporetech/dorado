#include "locale_utils.h"

#ifndef _WIN32
#include <stdlib.h>

#include <clocale>
#endif

namespace dorado::utils {

void ensure_user_locale_may_be_set() {
    // JIRA issue DOR-234.
    // The indicators library tries to set the user preferred locale, so we ensure
    // this is possible.
#ifdef _WIN32
    // No-op on windows as setlocale with an empty locale will succeed as it
    // will set the locale name to the value returned by GetUserDefaultLocaleName
    // whose only failure mode is ERROR_INSUFFICIENT_BUFFER, which will not be the case.
#else
    // An invalid value for the LANG environment variable (e.g. if it's not present
    // on the machine) may cause setting the user preferred locale with
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
#endif
}

}  // namespace dorado::utils
