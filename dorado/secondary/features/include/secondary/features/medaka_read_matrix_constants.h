#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace dorado::secondary {

// Value representing deletion in base channel.
static constexpr int8_t DEL_VAL = 5;

// convert 16bit IUPAC (+16 for strand) to plp_bases index
static constexpr std::array<int8_t, 32> NUM_TO_COUNT_BASE_SYMM{
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
        -1, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1,
};

// Name used in read id arrays where no read is present
static constexpr std::string_view BLANK_READ_PLACEHOLDER = "__blank";

}  // namespace dorado::secondary
