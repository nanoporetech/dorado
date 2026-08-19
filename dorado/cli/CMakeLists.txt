add_subdirectory(cli_lib)

add_executable(dorado
    main.cpp
)

set_target_properties(dorado
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (MSVC)
    set_property(TARGET dorado APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
endif()

enable_warnings_as_errors(dorado)
check_linked_libs(dorado)

target_link_libraries(dorado
    dorado_cli
    dorado_licences
    dorado_licences_pod5
    dorado_utils
    dorado_version
    minimap2
)

install(
    TARGETS dorado
    RUNTIME_DEPENDENCY_SET dorado_deps
    RUNTIME
        DESTINATION bin
        COMPONENT standalone_dorado
)

if(WIN32 AND TARGET pod5_format AND NOT POD5_STATIC)
    install(
        IMPORTED_RUNTIME_ARTIFACTS pod5_format
        RUNTIME_DEPENDENCY_SET dorado_deps
        RUNTIME
            DESTINATION bin
            COMPONENT redist_libs
    )
endif()

# Install any library dependencies
install(
    RUNTIME_DEPENDENCY_SET dorado_deps
    # despite what the docs would have you believe,
    # regexes need to come before artefact type
    # keywords like RUNTIME or LIBRARY
    PRE_EXCLUDE_REGEXES
        # api-* and ext-* are "Windows API sets" - virtual
        # aliases for system libraries, not real DLLs
        # https://learn.microsoft.com/en-us/windows/win32/apiindex/windows-apisets
        [[api-.*]]
        [[ext-.*]]
        # The dependency walker picks up some injected DLLs from
        # Windows Defender and such, so ignore those
        [[hvsifiletrust\.dll]]
        [[pdmutilities\.dll]]
        [[wpaxholder\.dll]]
    POST_EXCLUDE_REGEXES
        # Make sure we don't try to bundle system DLLs on Windows
        [[.*(\\|/)system32(\\|/).*\.dll]]
        # or on Linux
        [[^/(lib.*|usr)/.*]]
    DIRECTORIES
        # Because binaries are scattered all over, they're often
        # not in the same directory as the DLLs (on Windows),
        # which then means the lookup fails to find them.
        "${TORCH_LIB}/lib"
        "${HTSLIB_DIR}"
    RUNTIME
        DESTINATION bin
        COMPONENT redist_libs
    LIBRARY
        DESTINATION lib
        COMPONENT redist_libs
)
