#pragma once

#include <string>

namespace dorado::utils {

/**
 * @brief Generates a derived UUID from a given input UUID and a description string.
 *
 * This function takes an input UUID and a description string, concatenates them, and creates a new
 * UUID based on their SHA-256 hash. The resulting UUID has a version of 4 (random) and follows the
 * RFC 4122 variant.
 *
 * @param input_uuid The input UUID string to be used as the basis for generating the new UUID.
 * @param desc The description string to be combined with the input UUID for generating the new UUID.
 *
 * @return A derived UUID string based on the concatenation of the input_uuid and desc.
 *
 * Example:
 *   std::string input_uuid = "550e8400-e29b-41d4-a716-446655440000";
 *   std::string desc = "example_description";
 *   std::string derived_uuid = derive_uuid(input_uuid, desc);
 */
std::string derive_uuid(const std::string& input_uuid, const std::string& desc);

}  // namespace dorado::utils
