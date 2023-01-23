
set(SOURCE_FILES
    main.cpp
    Fast5DataLoaderTest.cpp
    TensorUtilsTest.cpp
    MathUtilsTest.cpp
    ReadTest.cpp
    RemoraEncoderTest.cpp
    SequenceUtilsTest.cpp
    StitchTest.cpp
    StereoDuplexTest.cpp
)

if(APPLE)
    list(APPEND SOURCE_FILES
        MetalLinearTest.cpp
    )
endif()

add_executable(dorado_tests ${SOURCE_FILES})

target_link_libraries(dorado_tests
    dorado_lib
    ${ZLIB_LIBRARIES}
)

add_test(
    NAME dorado_tests
    COMMAND dorado_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
