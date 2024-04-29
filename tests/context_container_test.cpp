#include "read_pipeline/context_container.h"

// libtorch defines a CHECK macro, but we want catch2's version for testing
#undef CHECK
#include <catch2/catch.hpp>

#define CUT_TAG "[dorado::ContextContainer]"

namespace {

struct SomeClass {};

struct SomeBaseClass {};

struct SomeDerivedClass : public SomeBaseClass {};

}  // namespace

namespace dorado::context_container::test {

TEST_CASE(CUT_TAG " constructor does not throw", CUT_TAG) { REQUIRE_NOTHROW(ContextContainer()); }

TEST_CASE(CUT_TAG " register_context() with T as T does not throw", CUT_TAG) {
    ContextContainer cut{};
    REQUIRE_NOTHROW(cut.register_context<SomeClass>(std::make_shared<SomeClass>()));
}

TEST_CASE(CUT_TAG " get_ptr() with class T registered as T does not throw", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<SomeClass>(std::make_shared<SomeClass>());

    REQUIRE_NOTHROW(cut.get_ptr<SomeClass>());
}

TEST_CASE(CUT_TAG " get_ptr() with class T registered as T returns the same instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto actual_instance = cut.get_ptr<SomeClass>();

    REQUIRE(actual_instance.get() == original_instance.get());
}

TEST_CASE(CUT_TAG " get_ptr() with default and match returns match instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto actual_instance = cut.get_ptr<SomeClass>(nullptr);

    REQUIRE(actual_instance.get() == original_instance.get());
}

TEST_CASE(CUT_TAG " get_ptr() with default and no match returns default", CUT_TAG) {
    ContextContainer cut{};

    REQUIRE(cut.get_ptr<SomeClass>(nullptr) == nullptr);
}

TEST_CASE(CUT_TAG " get() with class T registered as T returns the same instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto& actual_instance = cut.get<SomeClass>();

    REQUIRE(&actual_instance == original_instance.get());
}

TEST_CASE(CUT_TAG " get() const with class T registered as T returns the same instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& actual_instance = cut_const.get<SomeClass>();

    REQUIRE(&actual_instance == original_instance.get());
}

TEST_CASE(CUT_TAG " register_context() with T as T and T as superclass does not throw", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    REQUIRE_NOTHROW(cut.register_context<SomeBaseClass>(original_instance));
    REQUIRE_NOTHROW(cut.register_context<SomeDerivedClass>(original_instance));
}

TEST_CASE(CUT_TAG " get_ptr() with T when registered as T and as superclass of T", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto instance_as_derived = cut.get_ptr<SomeDerivedClass>();

    REQUIRE(instance_as_derived.get() == original_instance.get());
}

TEST_CASE(CUT_TAG " get() with T when registered as T and as superclass of T", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto& instance_as_derived = cut.get<SomeDerivedClass>();

    REQUIRE(&instance_as_derived == original_instance.get());
}

TEST_CASE(CUT_TAG " get() const with T when registered as T and as superclass of T", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& instance_as_derived = cut_const.get<SomeDerivedClass>();

    REQUIRE(&instance_as_derived == original_instance.get());
}

TEST_CASE(CUT_TAG " get_ptr() with superclass T when registered as T and as superclass of T",
          CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto instance_as_base = cut.get_ptr<SomeBaseClass>();

    REQUIRE(instance_as_base.get() == original_instance.get());
}

TEST_CASE(CUT_TAG " get() with superclass of T when registered as T and as superclass of T",
          CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto& instance_as_base = cut.get<SomeBaseClass>();

    REQUIRE(&instance_as_base == original_instance.get());
}

TEST_CASE(CUT_TAG " get() const with superclass of T when registered as T and as superclass of T",
          CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& instance_as_base = cut_const.get<SomeBaseClass>();

    REQUIRE(&instance_as_base == original_instance.get());
}

}  // namespace dorado::context_container::test
