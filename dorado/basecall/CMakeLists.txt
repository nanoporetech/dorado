add_subdirectory(benchmarks)

dorado_add_library(dorado_basecall
    NAME
        dorado_basecall
    PUBLIC_DIR
        basecall
    SOURCES_PUBLIC
        crf_utils.h
        CudaCaller.h
        CudaModelRunner.h
        DecodedChunk.h
        DecodedChunk.h
        MetalCaller.h
        MetalModelRunner.h
        ModelRunner.h
        ModelRunnerBase.h
    SOURCES_PRIVATE
        crf_utils.cpp
        decode/beam_search.cpp
        decode/beam_search.h
        decode/CPUDecoder.cpp
        decode/CPUDecoder.h
        decode/CUDADecoder.h
        decode/Decoder.cpp
        decode/Decoder.h
        model/CRFModel.cpp
        model/CRFModel.h
        model/MetalCRFModel.h
        model/TxModel.cpp
        model/TxModel.h
        ModelRunner.cpp
        ModelRunnerBase.cpp
    DEPENDS_PUBLIC
        dorado_config
        dorado_nn
        dorado_torch_utils
        dorado_utils
        torch_lib
    DEPENDS_PRIVATE
        spdlog::spdlog
)

if(APPLE)
    target_sources(dorado_basecall
        PRIVATE
            MetalCaller.cpp
            MetalModelRunner.cpp
            model/MetalCRFModel.cpp
    )
else()
    target_sources(dorado_basecall
        PRIVATE
            CudaCaller.cpp
            CudaModelRunner.cpp
            decode/CUDADecoder.cpp
    )
    target_link_libraries(dorado_basecall PRIVATE dorado_benchmarks)
endif()

if (TARGET koi)
    target_link_libraries(dorado_basecall PRIVATE koi)
endif()
