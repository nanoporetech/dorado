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
#   )
#
function(dorado_add_library)
    # Parse the args.
    set(options NO_TORCH)
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

    # Reuse the PCH if it makes use of torch.
    if (DORADO_ENABLE_PCH AND NOT arg_NO_TORCH)
        target_link_libraries(${arg_NAME} PRIVATE dorado_pch)
        target_precompile_headers(${arg_NAME} REUSE_FROM dorado_pch)
    endif()
endfunction()
