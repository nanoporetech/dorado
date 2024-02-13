#include "crypto_utils.h"

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#else
#include <openssl/sha.h>
#endif

namespace dorado::utils::crypto {

SHA256Digest sha256(std::string_view data) {
    // Digest the data.
    SHA256Digest hash{};
#if defined(__APPLE__)
    static_assert(std::size(SHA256Digest{}) == CC_SHA256_DIGEST_LENGTH);
    ::CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), hash.data());
#else
    static_assert(std::size(SHA256Digest{}) == SHA256_DIGEST_LENGTH);
    ::SHA256(reinterpret_cast<const unsigned char *>(data.data()), data.size(), hash.data());
#endif
    return hash;
}

}  // namespace dorado::utils::crypto
