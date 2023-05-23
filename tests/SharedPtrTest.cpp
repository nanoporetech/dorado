#include "utils/ns_shared_ptr.h"

#include <catch2/catch.hpp>

namespace {

#define DEFINE_TEST(name) TEST_CASE("SharedPtr: " name, "[SharedPtr]")

struct MockNSObject {
    int ref_count = 1;

    void retain() { ref_count++; }
    void release() { ref_count--; }
};

DEFINE_TEST("TransferPtr transfers ownership") {
    MockNSObject object;
    {
        auto ptr = dorado::utils::TransferPtr(&object);
        REQUIRE(object.ref_count == 1);
        REQUIRE(ptr.get() == &object);
    }
    REQUIRE(object.ref_count == 0);
}

DEFINE_TEST("RetainPtr shares ownership") {
    MockNSObject object;
    {
        auto ptr = dorado::utils::RetainPtr(&object);
        REQUIRE(object.ref_count == 2);
        REQUIRE(ptr.get() == &object);
    }
    REQUIRE(object.ref_count == 1);
}

DEFINE_TEST("Empty move/copy doesn't crash") {
    dorado::utils::SharedPtr<MockNSObject> ptr1, ptr2;
    ptr1 = ptr2;
    ptr2 = std::move(ptr1);
}

DEFINE_TEST("Copying increases ref count") {
    MockNSObject object;
    {
        auto ptr = dorado::utils::TransferPtr(&object);
        REQUIRE(object.ref_count == 1);

        // Copy ctor
        {
            dorado::utils::SharedPtr ptr2(ptr);
            REQUIRE(object.ref_count == 2);
            REQUIRE(ptr);
            REQUIRE(ptr2);
        }
        REQUIRE(object.ref_count == 1);

        // Copy assign
        {
            dorado::utils::SharedPtr<MockNSObject> ptr2;
            ptr2 = ptr;
            REQUIRE(object.ref_count == 2);
            REQUIRE(ptr);
            REQUIRE(ptr2);
        }
        REQUIRE(object.ref_count == 1);
    }
    REQUIRE(object.ref_count == 0);
}

DEFINE_TEST("Moving doesn't increase ref count") {
    MockNSObject object;

    // Move ctor
    auto ptr = dorado::utils::TransferPtr(&object);
    REQUIRE(object.ref_count == 1);
    {
        dorado::utils::SharedPtr ptr2(std::move(ptr));
        REQUIRE(object.ref_count == 1);
        REQUIRE_FALSE(ptr);
        REQUIRE(ptr2);
    }
    REQUIRE(object.ref_count == 0);

    // Move assign
    ptr = dorado::utils::RetainPtr(&object);
    REQUIRE(object.ref_count == 1);
    {
        dorado::utils::SharedPtr<MockNSObject> ptr2;
        ptr2 = std::move(ptr);
        REQUIRE(object.ref_count == 1);
        REQUIRE_FALSE(ptr);
        REQUIRE(ptr2);
    }
    REQUIRE(object.ref_count == 0);
}

}  // namespace
