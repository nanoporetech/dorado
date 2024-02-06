#include "crypto_utils.h"

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
static constexpr auto kDigestLength = CC_SHA256_DIGEST_LENGTH;
#else
#include <openssl/sha.h>
static constexpr auto kDigestLength = SHA256_DIGEST_LENGTH;
#endif

namespace dorado::utils::crypto {

SHA256Digest sha256(std::string_view data) {
    // Make sure we match the size of the API we're using.
    static_assert(std::size(SHA256Digest{}) == kDigestLength);
    static_assert(sizeof(SHA256Digest) == kDigestLength);

    // Digest the data.
    SHA256Digest hash{};
#if defined(__APPLE__)
    ::CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), hash.data());
#else
    ::SHA256(reinterpret_cast<const unsigned char *>(data.data()), data.size(), hash.data());
#endif
    return hash;
}

}  // namespace dorado::utils::crypto
