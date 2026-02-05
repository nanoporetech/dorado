dorado_add_library(
    NAME
        dorado_alignment
    PUBLIC_DIR
        alignment
    SOURCES_PUBLIC
        alignment_info.h
        bed_file.h
        BedFileAccess.h
        IndexFileAccess.h
        minimap2_args.h
        minimap2_wrappers.h
        Minimap2Index.h
        Minimap2IndexSupportTypes.h
        Minimap2MemoryLimiter.h
        Minimap2Options.h
        sam_utils.h # only public for unit tests
    SOURCES_PRIVATE
        bed_file.cpp
        BedFileAccess.cpp
        IndexFileAccess.cpp
        minimap2_args.cpp
        Minimap2Index.cpp
        Minimap2MemoryLimiter.cpp
        Minimap2Options.cpp
        sam_utils.cpp
    DEPENDS_PUBLIC
        htslib
        minimap2
        dorado_hts_utils
        dorado_utils
    DEPENDS_PRIVATE
        spdlog::spdlog
        argparse::argparse
    # Minimap2Aligner.cpp was moved to another target to allow
    # this one to be torch-less.
    NO_TORCH
)
