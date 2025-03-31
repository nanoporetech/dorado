#include "read_pipeline/context_container.h"

// libtorch defines a CHECK macro, but we want catch2's version for testing
#include <catch2/catch_test_macros.hpp>

#define CUT_TAG "[dorado::ContextContainer]"

namespace {

struct SomeClass {
    int value;
};

struct SomeBaseClass {};

struct SomeDerivedClass : public SomeBaseClass {};

}  // namespace

namespace dorado::context_container::test {

CATCH_TEST_CASE(CUT_TAG " constructor does not throw", CUT_TAG) {
    CATCH_REQUIRE_NOTHROW(ContextContainer());
}

CATCH_TEST_CASE(CUT_TAG " register_context() with T as T does not throw", CUT_TAG) {
    ContextContainer cut{};
    CATCH_REQUIRE_NOTHROW(cut.register_context<SomeClass>(std::make_shared<SomeClass>()));
}

CATCH_TEST_CASE(CUT_TAG " get_ptr() with class T registered as T does not throw", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE_NOTHROW(cut.get_ptr<SomeClass>());
}

CATCH_TEST_CASE(CUT_TAG " get_ptr() with class T registered as T returns the same instance",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto actual_instance = cut.get_ptr<SomeClass>();

    CATCH_REQUIRE(actual_instance.get() == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get() with class T registered as T returns the same instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto& actual_instance = cut.get<SomeClass>();

    CATCH_REQUIRE(&actual_instance == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get() const with class T registered as T returns the same instance",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& actual_instance = cut_const.get<SomeClass>();

    CATCH_REQUIRE(&actual_instance == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " register_context() with T as T and T as superclass does not throw",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    CATCH_REQUIRE_NOTHROW(cut.register_context<SomeBaseClass>(original_instance));
    CATCH_REQUIRE_NOTHROW(cut.register_context<SomeDerivedClass>(original_instance));
}

CATCH_TEST_CASE(CUT_TAG " get_ptr() with T when registered as T and as superclass of T", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto instance_as_derived = cut.get_ptr<SomeDerivedClass>();

    CATCH_REQUIRE(instance_as_derived.get() == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get() with T when registered as T and as superclass of T", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto& instance_as_derived = cut.get<SomeDerivedClass>();

    CATCH_REQUIRE(&instance_as_derived == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get() const with T when registered as T and as superclass of T",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& instance_as_derived = cut_const.get<SomeDerivedClass>();

    CATCH_REQUIRE(&instance_as_derived == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get_ptr() with superclass T when registered as T and as superclass of T",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto instance_as_base = cut.get_ptr<SomeBaseClass>();

    CATCH_REQUIRE(instance_as_base.get() == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get() with superclass of T when registered as T and as superclass of T",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    auto& instance_as_base = cut.get<SomeBaseClass>();

    CATCH_REQUIRE(&instance_as_base == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG
                " get() const with superclass of T when registered as T and as superclass of T",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeDerivedClass>();
    cut.register_context<SomeBaseClass, SomeDerivedClass>(original_instance);
    cut.register_context<SomeDerivedClass>(original_instance);

    const ContextContainer& cut_const = cut;
    auto& instance_as_base = cut_const.get<SomeBaseClass>();

    CATCH_REQUIRE(&instance_as_base == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " 'get_ptr<T>()' when T has not been registered returns nullptr", CUT_TAG) {
    ContextContainer cut{};

    CATCH_REQUIRE(cut.get_ptr<SomeClass>() == nullptr);
}

CATCH_TEST_CASE(CUT_TAG " 'get_ptr<T>()' when const T has been registered returns nullptr",
                CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<const SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE(cut.get_ptr<SomeClass>() == nullptr);
}

CATCH_TEST_CASE(CUT_TAG
                " 'get_ptr<const T>()' when T has been registered returns the same instance",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto actual_instance = cut.get_ptr<const SomeClass>();

    CATCH_REQUIRE(actual_instance.get() == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG
                " 'get_ptr<const T>()' when const T has been registered returns the same instance",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<const SomeClass>(original_instance);

    auto actual_instance = cut.get_ptr<const SomeClass>();

    CATCH_REQUIRE(actual_instance.get() == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get<T>() when T is not registered throws out_of_range", CUT_TAG) {
    ContextContainer cut{};

    CATCH_REQUIRE_THROWS_AS(cut.get<SomeClass>(), std::out_of_range);
}

CATCH_TEST_CASE(CUT_TAG " get<T>() when const T is registered throws out_of_range", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<const SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE_THROWS_AS(cut.get<SomeClass>(), std::out_of_range);
}

CATCH_TEST_CASE(CUT_TAG " get<const T>() when const T is registered returns the same instance",
                CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<const SomeClass>(original_instance);

    auto& actual_instance = cut.get<const SomeClass>();

    CATCH_REQUIRE(&actual_instance == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get<const T>() when T is registered returns the same instance", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);

    auto& actual_instance = cut.get<const SomeClass>();

    CATCH_REQUIRE(&actual_instance == original_instance.get());
}

CATCH_TEST_CASE(CUT_TAG " get<T>() when T is registered returns a non const reference", CUT_TAG) {
    ContextContainer cut{};
    auto original_instance = std::make_shared<SomeClass>();
    cut.register_context<SomeClass>(original_instance);
    constexpr int NEW_VALUE{42};
    CATCH_CHECK(original_instance->value != NEW_VALUE);

    auto& retrieved_instance = cut.get<SomeClass>();
    retrieved_instance.value = NEW_VALUE;

    CATCH_REQUIRE(original_instance->value == NEW_VALUE);
}

CATCH_TEST_CASE(CUT_TAG " exists<T>() when T is registered returns true", CUT_TAG) {
    ContextContainer cut{};

    cut.register_context<SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE(cut.exists<SomeClass>());
}

CATCH_TEST_CASE(CUT_TAG " exists<T>() when T is not registered returns false", CUT_TAG) {
    ContextContainer cut{};

    CATCH_REQUIRE_FALSE(cut.exists<SomeClass>());
}

CATCH_TEST_CASE(CUT_TAG " exists<T>() when const T is registered returns false", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<const SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE_FALSE(cut.exists<SomeClass>());
}

CATCH_TEST_CASE(CUT_TAG " exists<const T>() when const T is registered returns true", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<const SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE(cut.exists<const SomeClass>());
}

CATCH_TEST_CASE(CUT_TAG " exists<const T>() when T is registered returns true", CUT_TAG) {
    ContextContainer cut{};
    cut.register_context<SomeClass>(std::make_shared<SomeClass>());

    CATCH_REQUIRE(cut.exists<const SomeClass>());
}

}  // namespace dorado::context_container::test
