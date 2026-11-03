#include "alignment/minimap2_args.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#define CUT_TAG "[split_attached_u_values]"

using dorado::alignment::mm2::split_attached_u_values;

CATCH_TEST_CASE(CUT_TAG " attached value is split", CUT_TAG) {
    std::vector<std::string> args{"prog", "-uf"};
    split_attached_u_values(args);
    CATCH_CHECK(args == std::vector<std::string>{"prog", "-u", "f"});
}

CATCH_TEST_CASE(CUT_TAG " separate value is unchanged", CUT_TAG) {
    std::vector<std::string> args{"prog", "-u", "f"};
    split_attached_u_values(args);
    CATCH_CHECK(args == std::vector<std::string>{"prog", "-u", "f"});
}

CATCH_TEST_CASE(CUT_TAG " unrelated args are unchanged", CUT_TAG) {
    std::vector<std::string> args{"prog", "-k", "15", "--junc-bed", "-x", "splice"};
    const auto expected = args;
    split_attached_u_values(args);
    CATCH_CHECK(args == expected);
}

CATCH_TEST_CASE(CUT_TAG " empty args", CUT_TAG) {
    std::vector<std::string> args;
    split_attached_u_values(args);
    CATCH_CHECK(args.empty());
}

CATCH_TEST_CASE(CUT_TAG " multiple and mixed args", CUT_TAG) {
    std::vector<std::string> args{"prog", "-k", "15", "-ub", "-Y", "-un"};
    split_attached_u_values(args);
    CATCH_CHECK(args == std::vector<std::string>{"prog", "-k", "15", "-u", "b", "-Y", "-u", "n"});
}

CATCH_TEST_CASE(CUT_TAG " invalid attached value is left for the parser", CUT_TAG) {
    std::vector<std::string> args{"prog", "-uxyz"};
    split_attached_u_values(args);
    CATCH_CHECK(args == std::vector<std::string>{"prog", "-u", "xyz"});
}

CATCH_TEST_CASE(CUT_TAG " invalid missing value is left for the parser", CUT_TAG) {
    std::vector<std::string> args{"prog", "-u"};
    split_attached_u_values(args);
    CATCH_CHECK(args == std::vector<std::string>{"prog", "-u"});
}