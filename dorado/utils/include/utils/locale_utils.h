#pragma once

namespace dorado::utils {
// Ensures std::setlocale(LC_ALL, "") will succeed.
// See DOR-234 for details.
void ensure_user_locale_may_be_set();
}  // namespace dorado::utils