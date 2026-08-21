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

dorado_install_runtime_dependencies(dorado_deps standalone_dorado)

if(WIN32 AND TARGET pod5_format AND NOT POD5_STATIC)
    install(
        IMPORTED_RUNTIME_ARTIFACTS pod5_format
        RUNTIME_DEPENDENCY_SET dorado_deps
        RUNTIME
            DESTINATION bin
            COMPONENT standalone_dorado
    )
endif()
