set(SOURCE_FILES 
    main.cpp
    Fast5DataLoaderTest.cpp
    ReadTest.cpp
    RemoraEncoderTest.cpp
    StitchTest.cpp
    )

add_executable(dorado_tests ${SOURCE_FILES})

target_link_libraries(dorado_tests dorado_lib)

add_test(
    NAME dorado_tests
    COMMAND dorado_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)