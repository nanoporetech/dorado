#include "alignment_utils.h"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace dorado::utils {

std::string derive_uuid(const std::string& input_uuid, const std::string& desc) {
    // Hash the input UUID using SHA-256
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, input_uuid.c_str(), input_uuid.size());
    SHA256_Update(&sha256, desc.c_str(), desc.size());
    SHA256_Final(hash, &sha256);

    // Truncate the hash to 16 bytes (128 bits) to match the size of a UUID
    std::array<unsigned char, 16> truncated_hash;
    std::copy(std::begin(hash), std::begin(hash) + 16, std::begin(truncated_hash));

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
