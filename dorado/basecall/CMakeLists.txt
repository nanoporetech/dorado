add_subdirectory(benchmarks)

add_library(dorado_basecall STATIC
    # Public
    include/basecall/crf_utils.h
    include/basecall/CudaCaller.h
    include/basecall/CudaModelRunner.h
    include/basecall/DecodedChunk.h
    include/basecall/DecodedChunk.h
    include/basecall/MetalCaller.h
    include/basecall/MetalModelRunner.h
    include/basecall/ModelRunner.h
    include/basecall/ModelRunnerBase.h

    # Private
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

target_link_libraries(dorado_basecall
    PUBLIC
        dorado_config
        dorado_nn
        dorado_torch_utils
        dorado_utils
        torch_lib
    PRIVATE
        spdlog::spdlog
)

if (TARGET koi)
    target_link_libraries(dorado_basecall PRIVATE koi)
endif()

target_include_directories(dorado_basecall
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_warnings_as_errors(dorado_basecall)

if (DORADO_ENABLE_PCH)
    target_precompile_headers(dorado_basecall REUSE_FROM dorado_torch_utils)
endif()
