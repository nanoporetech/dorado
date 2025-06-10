add_library(dorado_read_pipeline_nodes STATIC
    # Public
    include/read_pipeline/nodes/AdapterDetectorNode.h
    include/read_pipeline/nodes/AlignerNode.h
    include/read_pipeline/nodes/BarcodeClassifierNode.h
    include/read_pipeline/nodes/BarcodeDemuxerNode.h
    include/read_pipeline/nodes/BasecallerNode.h
    include/read_pipeline/nodes/BaseSpaceDuplexCallerNode.h
    include/read_pipeline/nodes/CorrectionInferenceNode.h
    include/read_pipeline/nodes/CorrectionPafWriterNode.h
    include/read_pipeline/nodes/DuplexReadTaggingNode.h
    include/read_pipeline/nodes/HtsWriterNode.h
    include/read_pipeline/nodes/ModBaseCallerNode.h
    include/read_pipeline/nodes/ModBaseChunkCallerNode.h
    include/read_pipeline/nodes/NullNode.h
    include/read_pipeline/nodes/PairingNode.h
    include/read_pipeline/nodes/PolyACalculatorNode.h
    include/read_pipeline/nodes/ReadFilterNode.h
    include/read_pipeline/nodes/ReadForwarderNode.h
    include/read_pipeline/nodes/ReadSplitNode.h
    include/read_pipeline/nodes/ReadToBamTypeNode.h
    include/read_pipeline/nodes/ScalerNode.h
    include/read_pipeline/nodes/StereoDuplexEncoderNode.h
    include/read_pipeline/nodes/SubreadTaggerNode.h
    include/read_pipeline/nodes/TrimmerNode.h

    # Private
    AdapterDetectorNode.cpp
    AlignerNode.cpp
    BarcodeClassifierNode.cpp
    BarcodeDemuxerNode.cpp
    BasecallerNode.cpp
    BaseSpaceDuplexCallerNode.cpp
    CMakeLists.txt
    CorrectionInferenceNode.cpp
    CorrectionPafWriterNode.cpp
    DuplexReadTaggingNode.cpp
    HtsWriterNode.cpp
    ModBaseCallerNode.cpp
    ModBaseChunkCallerNode.cpp
    NullNode.cpp
    PairingNode.cpp
    PolyACalculatorNode.cpp
    ReadFilterNode.cpp
    ReadForwarderNode.cpp
    ReadSplitNode.cpp
    ReadToBamTypeNode.cpp
    ScalerNode.cpp
    StereoDuplexEncoderNode.cpp
    SubreadTaggerNode.cpp
    TrimmerNode.cpp
)

target_link_libraries(dorado_read_pipeline_nodes
    PUBLIC
        dorado_alignment
        dorado_read_pipeline_base
        dorado_utils
    PRIVATE
        cxxpool
        dorado_basecall
        dorado_correct
        dorado_demux
        dorado_modbase
        dorado_poly_tail
        dorado_splitter
        spdlog::spdlog
)

target_include_directories(dorado_read_pipeline_nodes
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_read_pipeline_nodes)
