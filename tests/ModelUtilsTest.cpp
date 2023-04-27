#include "utils/models.h"

#include <catch2/catch.hpp>

#define TEST_TAG "[ModelUtils]"

TEST_CASE(TEST_TAG " Get model sample rate by name") {
    SECTION("Check against valid 5khz model name") {
        CHECK(dorado::utils::get_sample_rate_by_model_name(
                      "dna_r10.4.1_e8.2_5khz_400bps_fast@v4.2.0") == 5000);
    }
    SECTION("Check against valid 4khz model name") {
        CHECK(dorado::utils::get_sample_rate_by_model_name("dna_r10.4.1_e8.2_260bps_fast@v4.0.0") ==
              4000);
    }
    SECTION("Check against unknown model name") {
        CHECK(dorado::utils::get_sample_rate_by_model_name("blah") == 4000);
    }
}
