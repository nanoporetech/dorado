# Helper to create a cmake library target.
#
# Example usage:
#
#   dorado_add_library(
#       NAME
#           dorado_example
#       PUBLIC_DIR
#           example
#       SOURCES_PUBLIC
#           blah.h # lives at include/example/blah.h
#       SOURCES_PRIVATE
#           blah.cpp
#           internal.h # implementation detail, not intended for dependents to include
#       SOURCES_TEST
#           test.cpp # testing code has access to private sources too
#       DEPENDS_PUBLIC
#           dorado_torch_utils # blah.h includes torch and hence depends publicly on it
#       DEPENDS_PRIVATE
#           toml11::toml11 # implementation uses toml11 but dependents don't need to know that
#       NO_TORCH # optionally can be added to say that this target doesn't need the torch pch
#   )
#
function(dorado_add_library)
    # Parse the args.
    set(options NO_TORCH)
    set(oneValueArgs NAME PUBLIC_DIR)
    set(multiValueArgs SOURCES_PUBLIC SOURCES_PRIVATE SOURCES_TEST DEPENDS_PUBLIC DEPENDS_PRIVATE)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create the library.
    add_library(${arg_NAME} STATIC)
    foreach (src ${arg_SOURCES_PUBLIC})
        # All public sources must exist in the public include dir.
        target_sources(${arg_NAME} PUBLIC include/${arg_PUBLIC_DIR}/${src})
    endforeach()
    target_sources(${arg_NAME} PRIVATE ${arg_SOURCES_PRIVATE})
    set(targets ${arg_NAME})

    # Create a test executable if tests have been provided.
    if (DEFINED arg_SOURCES_TEST AND NOT DORADO_DISABLE_TESTS)
        set(test_name test_${arg_NAME})
        add_executable(${test_name})
        foreach (src ${arg_SOURCES_TEST})
            # All test files must exist in the test dir.
            target_sources(${test_name} PUBLIC tests/${src})
        endforeach()
        target_link_libraries(${test_name} PRIVATE ${arg_NAME} dorado_tests_common)
        dorado_add_test(${test_name})
        list(APPEND targets ${test_name})

        # Include this test in the coverage report.
        if (GENERATE_TEST_COVERAGE)
            add_dependencies(dorado_test_coverage ${test_name})
        endif()
    endif()

    # Validate that this target really doesn't link to torch.
    # We defer this so that links not using this helper function aren't missed.
    # Note that we don't check the tests since dorado_test_common links to torch.
    if (arg_NO_TORCH)
        cmake_language(EVAL CODE "
            cmake_language(DEFER
                DIRECTORY ${CMAKE_SOURCE_DIR}
                CALL check_no_dependency_on_torch [[${arg_NAME}]]
            )
        ")
    endif()

    foreach (target IN LISTS targets)
        # Add our dependencies.
        target_link_libraries(${target}
            PUBLIC ${arg_DEPENDS_PUBLIC}
            PRIVATE ${arg_DEPENDS_PRIVATE}
        )

        # Anything linking to us can use our public includes only.
        target_include_directories(${target} PUBLIC include)

        # Anything internal can see what's inside.
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

        # All of our code should compile with warnings enabled.
        enable_warnings_as_errors(${target})

        # Reuse the PCH if it makes use of torch.
        if (DORADO_ENABLE_PCH AND NOT arg_NO_TORCH)
            get_target_property(type ${target} TYPE)
            if (type STREQUAL "EXECUTABLE")
                target_precompile_headers(${target} REUSE_FROM dorado_pch_exe)
            else()
                target_link_libraries(${target} PRIVATE dorado_pch)
                target_precompile_headers(${target} REUSE_FROM dorado_pch)
            endif()
        endif()

        # Enable coverage if told to do so.
        if (GENERATE_TEST_COVERAGE)
            append_coverage_compiler_flags_to_target(${target})
        endif()
    endforeach()
endfunction()

# Install runtime dependencies for an executable component.  Keep this in one
# place so the standalone binary and unit tests use identical dependency
# discovery and filtering rules.
function(dorado_install_runtime_dependencies DEPENDENCY_SET COMPONENT)
    install(
        RUNTIME_DEPENDENCY_SET ${DEPENDENCY_SET}
        # Despite what the docs would have you believe, regexes need to come
        # before artefact-type keywords like RUNTIME or LIBRARY.
        PRE_EXCLUDE_REGEXES
            # api-* and ext-* are virtual Windows API-set aliases.
            [[api-.*]]
            [[ext-.*]]
            # Ignore DLLs injected by Windows Defender and similar tools.
            [[hvsifiletrust\.dll]]
            [[pdmutilities\.dll]]
            [[wpaxholder\.dll]]
        POST_EXCLUDE_REGEXES
            # Do not bundle system libraries.
            [[.*(\\|/)system32(\\|/).*\.dll]]
            [[^/(lib.*|usr)/.*]]
        DIRECTORIES
            "${TORCH_LIB}/lib"
            "${HTSLIB_DIR}"
        RUNTIME
            DESTINATION bin
            COMPONENT ${COMPONENT}
        LIBRARY
            DESTINATION lib
            COMPONENT ${COMPONENT}
    )
endfunction()

# Wrapper for add_test(), plus some extra bits that we need.
function(dorado_add_test TARGET)
    # Keep an aggregate target for CI jobs which build only the unit tests.
    add_dependencies(dorado_unit_tests ${TARGET})
    set_property(GLOBAL APPEND PROPERTY DORADO_UNIT_TEST_TARGETS ${TARGET})

    # Add the test.
    add_test(
        NAME ${TARGET}
        COMMAND ${TARGET}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # The tests need to be able to find the libs in order to run.
    # We also want these libs to take priority over any installed on the system, so prepend them.
    if (MSVC)
        set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/bin")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/lib")
    endif()

    # For some reason when TSan is enabled on x64/Linux we get a failure to load dependencies of torch, so add it explicitly to the path
    if (ECM_ENABLE_SANITIZERS AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
        set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${TORCH_LIB}/lib")
    endif()

    # Enable the metal validation layer when running tests.
    if (APPLE)
        set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "MTL_DEBUG_LAYER=set:1")
        # We can't have warnings as errors because of the MPS backend in torch. Even enabling it in
        # logging mode is too spammy.
        #set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "MTL_DEBUG_LAYER_WARNING_MODE=set:nslog")

        # It would be nice to enable this fully for the simulator, however the simulator simulates old hardware
        # which we don't plan on supporting (256MB maxBufferLength, lack of compute memory barriers, etc...).
        #set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT_MODIFICATION "SIMCTL_CHILD_MTL_DEBUG_LAYER=set:1")
    endif()

    install(
        TARGETS ${TARGET}
        RUNTIME_DEPENDENCY_SET dorado_unit_test_deps
        DESTINATION bin
        COMPONENT unit_tests
        EXCLUDE_FROM_ALL
    )

    if(UNIX AND NOT APPLE)
        # GNU ld preserves a trailing empty RPATH entry only when the RPATH
        # begins with a non-empty entry.  Without this, CMake's install script
        # can expect an RPATH that differs from the linked ELF RUNPATH.
        set_property(TARGET ${TARGET} PROPERTY BUILD_RPATH "$ORIGIN")
    endif()

endfunction()

# Generate a standalone CTest project for the installed unit-test executables.
# This avoids duplicating the test-target list in CI while retaining the test
# working directory expected by the fixtures in the source checkout.
function(dorado_install_unit_test_ctest_project)
    get_property(unit_test_targets GLOBAL PROPERTY DORADO_UNIT_TEST_TARGETS)

    set(unit_test_ctest_contents [=[
cmake_minimum_required(VERSION 3.25)
project(dorado_unit_tests NONE)
enable_testing()
]=])
    foreach (target IN LISTS unit_test_targets)
        string(APPEND unit_test_ctest_contents
            "add_test(NAME ${target} COMMAND \"\${CMAKE_CURRENT_LIST_DIR}/../bin/$<TARGET_FILE_NAME:${target}>\")\n"
            "set_tests_properties(${target} PROPERTIES WORKING_DIRECTORY \"\${CMAKE_CURRENT_LIST_DIR}/../..\")\n"
        )
    endforeach()

    file(GENERATE
        OUTPUT "${CMAKE_BINARY_DIR}/unit_test_ctest/CMakeLists.txt"
        CONTENT "${unit_test_ctest_contents}"
    )
    install(
        FILES "${CMAKE_BINARY_DIR}/unit_test_ctest/CMakeLists.txt"
        DESTINATION tests
        COMPONENT unit_tests
    )
endfunction()
