#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

template <typename... Ts>
std::array<std::byte, sizeof...(Ts)> make_bytes(Ts&&... args) noexcept {
    return {std::byte(std::forward<Ts>(args))...};
}

template <typename T>
void decode_base64(const std::string& input, std::vector<T>& target) {
    // Base64 encoding converts any 3 bytes into 4 printable bytes by splitting them into four sets of 6-byte values
    //  corresponding to one of 64 printable characters 0-9, a-z, A-Z, + and /.  The mapping works as follows:
    // Binary bytes : |1 1 1 1 1 1 1 1|2 2 2 2 2 2 2 2|3 3 3 3 3 3 3 3|
    // 6-bit indices: |1 1 1 1 1 1|2 2 2 2 2 2|3 3 3 3 3 3|4 4 4 4 4 4|
    // The number of binary bytes to be emitted can be calculated from the number of input byte indices. If the size of
    //  the output array does not divide cleanly by 3, there will either be 2 input bytes to decode, or 3.  1 would not
    //  be valid, as 1 input 6-bit index cannot reconstruct a whole output byte.  Some encodings of Base64 will pad the
    //  printable string with '=' characters, so we trim them off before decoding.

    // Trim any padding '=' off the input
    size_t input_bytes = input.size();
    while (input[input_bytes - 1] == '=') {
        input_bytes--;
    }

    // Calculate how many bytes we should be emitting.
    size_t num_loops = input_bytes / 4;
    size_t remnant_bytes_in = input_bytes % 4;
    if (remnant_bytes_in == 1) {
        throw std::runtime_error("Incorrect remnant length in Base64 string!");
    }
    size_t remnant_bytes_out = remnant_bytes_in == 2 ? 1 : remnant_bytes_in == 3 ? 2 : 0;
    size_t total_bytes_out = num_loops * 3 + remnant_bytes_out;
    if (total_bytes_out % sizeof(T) != 0)
        throw std::runtime_error("Unaligned decoded length for data buffer!");

    // clang-format off
    static const auto lookup = make_bytes(
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, /*0-15*/
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, /*16-31*/
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  62, 0,  0,  0,  63, /*32-47*/ /* '+' and '/' */
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0,  0,  0,  0,  0,  0, /*48-63*/ /* '0' - '9' */
        0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,/*64-79*/ /* 'A'-'O' */
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 0,  0,  0,  0,  0, /*80-95*/ /* 'P' - 'Z' */
        0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,/*96-113*/ /* 'a' - 'q' */
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0,  0,  0,  0,  0  /*114-127*/ /* 'r' - 'z' */
    );
    // clang-format on

    size_t in_idx = 0;
    target.resize(total_bytes_out / sizeof(T));
    std::byte* out_ptr = reinterpret_cast<std::byte*>(target.data());
    for (size_t i = 0; i < num_loops; i++) {
        std::byte in1 = lookup[(input[in_idx++])];
        std::byte in2 = lookup[(input[in_idx++])];
        std::byte in3 = lookup[(input[in_idx++])];
        std::byte in4 = lookup[(input[in_idx++])];
        *out_ptr = (in1 << 2) | (in2 >> 4);  // 1st out byte
        out_ptr++;
        *out_ptr = (in2 << 4) | (in3 >> 2);  // 2nd out byte
        out_ptr++;
        *out_ptr = (in3 << 6) | in4;  // 3rd out byte
        out_ptr++;
    }

    // Decode the remnant
    if (remnant_bytes_in != 0) {
        std::byte in1 = lookup[(input[in_idx++])];
        std::byte in2 = lookup[(input[in_idx++])];
        *out_ptr = (in1 << 2) | (in2 >> 4);  // 1st out byte
        out_ptr++;
        if (remnant_bytes_in == 3) {
            std::byte in3 = lookup[(input[in_idx++])];
            *out_ptr = (in2 << 4) | (in3 >> 2);  // 2nd out byte
            out_ptr++;
        }
    }
}  // ----------------------------------------------------------------------------------------------

}  // namespace utils