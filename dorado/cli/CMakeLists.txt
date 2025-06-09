add_subdirectory(utils)

add_executable(dorado
    aligner.cpp
    basecall_output_args.cpp
    basecall_output_args.h
    basecaller.cpp
    cli.h
    correct.cpp
    demux.cpp
    download.cpp
    duplex.cpp
    main.cpp
    model_resolution.h
    polish.cpp
    summary.cpp
    trim.cpp
    variant.cpp
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

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado REUSE_FROM dorado_lib)
endif()

target_link_libraries(dorado
    cxxpool
    dorado_cli_utils
    dorado_compat
    dorado_file_info
    dorado_io_lib
    dorado_lib
    dorado_model_downloader_lib
    dorado_models_lib
    dorado_polish
    dorado_variant
    minimap2
)

install(TARGETS dorado RUNTIME)
