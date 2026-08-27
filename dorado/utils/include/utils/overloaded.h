#pragma once

namespace dorado::utils {

// Allows less ugliness in use of std::visit.
//
// Usage:
//
//   std::variant<int, float> variant = ...;
//
//   std::visit(utils::overloaded(
//     [](int i) { do_something(i); },
//     [](float f) { do_something_else(f); }
//   ), variant);
//

template <class... Ts>
// NOLINTNEXTLINE(misc-multiple-inheritance): we need multiple inheritance for this to work.
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace dorado::utils
