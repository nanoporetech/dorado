
set(SOURCE_FILES
    main.cpp
    AsyncQueueTest.cpp
    Fast5DataLoaderTest.cpp
    Pod5DataLoaderTest.cpp
    TensorUtilsTest.cpp
    MathUtilsTest.cpp
    ReadTest.cpp
    RemoraEncoderTest.cpp
    SequenceUtilsTest.cpp
    StitchTest.cpp
    StereoDuplexTest.cpp
    DuplexSplitTest.cpp
    TrimTest.cpp
    AlignerTest.cpp
    BamReaderTest.cpp
    BamWriterTest.cpp
    CliUtilsTest.cpp
    ReadFilterNodeTest.cpp
    ModelUtilsTest.cpp
    NodeSmokeTest.cpp
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        list(APPEND SOURCE_FILES
            MetalLinearTest.cpp
        )
    else()
        list(APPEND SOURCE_FILES
            cuda_utils_test.cpp
        )
    endif()
endif()

add_executable(dorado_tests ${SOURCE_FILES})

target_precompile_headers(dorado_tests REUSE_FROM dorado_lib)

target_link_libraries(dorado_tests
    dorado_lib
    dorado_io_lib
    dorado_models_lib
    minimap2
    ${ZLIB_LIBRARIES}
)

add_test(
    NAME dorado_tests
    COMMAND dorado_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
