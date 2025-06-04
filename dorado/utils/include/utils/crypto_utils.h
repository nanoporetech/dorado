#pragma once

#include <array>
#include <string_view>

namespace dorado::utils::crypto {

using SHA256Digest = std::array<unsigned char, 32>;
SHA256Digest sha256(std::string_view data);

}  // namespace dorado::utils::crypto
