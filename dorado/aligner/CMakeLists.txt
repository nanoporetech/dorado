# The reason this target is separate from dorado_alignment is so that
# the client in ont_core can link to dorado_alignment without linking
# to torch.
dorado_add_library(
    NAME
        dorado_aligner
    PUBLIC_DIR
        aligner
    SOURCES_PUBLIC
        Minimap2Aligner.h
    SOURCES_PRIVATE
        Minimap2Aligner.cpp
    DEPENDS_PUBLIC
        dorado_alignment
        dorado_read_pipeline_base
    DEPENDS_PRIVATE
        spdlog::spdlog
)
