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

install(TARGETS dorado RUNTIME)
