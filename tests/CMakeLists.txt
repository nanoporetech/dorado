
set(UNIT_TEST_SOURCE_FILES
    main.cpp
    AsyncQueueTest.cpp
    Fast5DataLoaderTest.cpp
    Pod5DataLoaderTest.cpp
    TensorUtilsTest.cpp
    MathUtilsTest.cpp
    ReadTest.cpp
    ModBaseEncoderTest.cpp
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
    PairingNodeTest.cpp
    PipelineTest.cpp
    BamUtilsTest.cpp
    ResumeLoaderTest.cpp
    TimeUtilsTest.cpp
    DuplexReadTaggingNodeTest.cpp
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        list(APPEND UNIT_TEST_SOURCE_FILES
	    MetalLinearTest.cpp
        )
    else()
        list(APPEND UNIT_TEST_SOURCE_FILES
            cuda_utils_test.cpp
        )
    endif()
endif()

add_executable(dorado_tests ${UNIT_TEST_SOURCE_FILES})

set(SMOKE_TEST_SOURCE_FILES
    main.cpp
    NodeSmokeTest.cpp
)

add_executable(dorado_smoke_tests ${SMOKE_TEST_SOURCE_FILES})

foreach(TEST_BIN dorado_tests dorado_smoke_tests)
    target_precompile_headers(${TEST_BIN} REUSE_FROM dorado_lib)

    target_link_libraries(${TEST_BIN}
        dorado_lib
        dorado_io_lib
        dorado_models_lib
        minimap2
        )

    target_include_directories(${TEST_BIN}
        SYSTEM
        PRIVATE
        ${DORADO_3RD_PARTY}/catch2
        )

    add_test(
        NAME ${TEST_BIN}
        COMMAND ${TEST_BIN}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )

    # The tests need to be able to find the libs in order to run.
    # We also want these libs to take priority over any installed on the system, so prepend them.
    if (MSVC)
        set_tests_properties(${TEST_BIN} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/bin")
        set_target_properties(${TEST_BIN} PROPERTIES LINK_OPTIONS "/ignore:4099")
    endif()

    # For some reason when TSan is enabled on x64/Linux we get a failure to load dependencies of torch, so add it explicitly to the path
    if ((ECM_ENABLE_SANITIZERS MATCHES "thread") AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
        set_tests_properties(${TEST_BIN} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${TORCH_LIB}/lib")
    endif()
endforeach()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
