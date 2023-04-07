
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
    AlignerTest.cpp
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
