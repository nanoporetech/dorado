#include "uuid_utils.h"

#include "crypto_utils.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace dorado::utils {

std::string derive_uuid(const std::string& input_uuid, const std::string& desc) {
    // Hash the input UUID+desc using SHA-256
    const auto hash = utils::crypto::sha256(input_uuid + desc);

    // Truncate the hash to 16 bytes (128 bits) to match the size of a UUID
    std::array<unsigned char, 16> truncated_hash;
    std::copy_n(std::begin(hash), truncated_hash.size(), std::begin(truncated_hash));

    // Set the UUID version to 4 (random)
    truncated_hash[6] = (truncated_hash[6] & 0x0F) | 0x40;

    // Set the UUID variant to the RFC 4122 specified value (10)
    truncated_hash[8] = (truncated_hash[8] & 0x3F) | 0x80;

    // Convert the truncated hash to a UUID string
    std::stringstream ss;
    for (size_t i = 0; i < truncated_hash.size(); ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(truncated_hash[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            ss << "-";
        }
    }

    return ss.str();
}

}  // namespace dorado::utils
