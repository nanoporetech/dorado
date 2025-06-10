add_library(dorado_alignment STATIC
    # Public
    include/alignment/alignment_info.h
    include/alignment/alignment_processing_items.h
    include/alignment/bed_file.h
    include/alignment/BedFileAccess.h
    include/alignment/IndexFileAccess.h
    include/alignment/minimap2_args.h
    include/alignment/minimap2_wrappers.h
    include/alignment/Minimap2Aligner.h
    include/alignment/Minimap2Index.h
    include/alignment/Minimap2IndexSupportTypes.h
    include/alignment/Minimap2Options.h
    include/alignment/sam_utils.h # only public for unit tests

    # Private
    alignment_processing_items.cpp
    bed_file.cpp
    BedFileAccess.cpp
    IndexFileAccess.cpp
    minimap2_args.cpp
    Minimap2Aligner.cpp
    Minimap2Index.cpp
    Minimap2Options.cpp
    sam_utils.cpp
)

target_link_libraries(dorado_alignment
    PUBLIC
        htslib
        minimap2
        dorado_utils
        dorado_read_pipeline_base
    PRIVATE
        spdlog::spdlog
)

target_include_directories(dorado_alignment
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_alignment)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_alignment REUSE_FROM dorado_pch)
endif()
