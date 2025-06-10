add_subdirectory(utils)

add_executable(dorado
    aligner.cpp
    basecall_output_args.cpp
    basecall_output_args.h
    basecaller.cpp
    cli.h
    correct.cpp
    CorrectionAligner.h
    CorrectionMapper.cpp
    CorrectionMapper.h
    CorrectionPafReader.cpp
    CorrectionPafReader.h
    CorrectionProgressTracker.cpp
    CorrectionProgressTracker.h
    demux.cpp
    download.cpp
    duplex.cpp
    main.cpp
    model_resolution.h
    polish_progress_tracker.cpp
    polish_progress_tracker.h
    polish.cpp
    ProgressTracker.cpp
    ProgressTracker.h
    read_output_progress_stats.cpp
    read_output_progress_stats.h
    summary.cpp
    trim.cpp
    variant_progress_tracker.cpp
    variant_progress_tracker.h
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
    target_precompile_headers(dorado REUSE_FROM dorado_pch)
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
    dorado_summary
    indicators
    minimap2
)

install(TARGETS dorado RUNTIME)
