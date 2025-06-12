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
#       DEPENDS_PUBLIC
#           dorado_torch_utils # blah.h includes torch and hence depends publicly on it
#       DEPENDS_PRIVATE
#           toml11::toml11 # implementation uses toml11 but dependents don't need to know that
#       NO_TORCH # optionally can be added to say that this target doesn't need the torch pch
#       POSITION_INDEPENDENT_CODE # optionally can be added for libs that go into shared objects
#   )
#
function(dorado_add_library)
    # Parse the args.
    set(options NO_TORCH POSITION_INDEPENDENT_CODE)
    set(oneValueArgs NAME PUBLIC_DIR)
    set(multiValueArgs SOURCES_PUBLIC SOURCES_PRIVATE DEPENDS_PUBLIC DEPENDS_PRIVATE)
    cmake_parse_arguments(arg "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create the library.
    add_library(${arg_NAME} STATIC)
    foreach (src ${arg_SOURCES_PUBLIC})
        # All public sources must exist in the public include dir.
        target_sources(${arg_NAME} PUBLIC include/${arg_PUBLIC_DIR}/${src})
    endforeach()
    target_sources(${arg_NAME} PRIVATE ${arg_SOURCES_PRIVATE})
    target_link_libraries(${arg_NAME}
        PUBLIC ${arg_DEPENDS_PUBLIC}
        PRIVATE ${arg_DEPENDS_PRIVATE}
    )

    # Anything linking to us can use our public includes only.
    target_include_directories(${arg_NAME} PUBLIC include)

    # All of our code should compile with warnings enabled.
    enable_warnings_as_errors(${arg_NAME})

    # Targets that are compiled into a dynamic library need to be PIC.
    if (arg_POSITION_INDEPENDENT_CODE)
        set_target_properties(${arg_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()

    # Reuse the PCH if it makes use of torch.
    if (arg_NO_TORCH)
        # Validate that this target really doesn't link to torch.
        # We defer this so that links not using this helper function aren't missed.
        cmake_language(EVAL CODE "
            cmake_language(DEFER
                DIRECTORY ${CMAKE_SOURCE_DIR}
                CALL check_no_dependency_on_torch [[${arg_NAME}]]
            )
        ")
    elseif (DORADO_ENABLE_PCH)
        target_link_libraries(${arg_NAME} PRIVATE dorado_pch)
        target_precompile_headers(${arg_NAME} REUSE_FROM dorado_pch)
    endif()

    # Enable coverage if told to do so.
    if (GENERATE_TEST_COVERAGE)
        append_coverage_compiler_flags_to_target(${arg_NAME})
    endif()
endfunction()
