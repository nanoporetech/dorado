
add_executable(dorado_tests
    main.cpp
    AdapterDetectorTest.cpp
    AlignerTest.cpp
    AsyncQueueTest.cpp
    BamReaderTest.cpp
    BamUtilsTest.cpp
    BamWriterTest.cpp
    BarcodeClassifierSelectorTest.cpp
    BarcodeClassifierTest.cpp
    BarcodeDemuxerNodeTest.cpp    
    CliUtilsTest.cpp
    CRFModelConfigTest.cpp
    CustomBarcodeParserTest.cpp
    DuplexReadTaggingNodeTest.cpp
    DuplexSplitTest.cpp
    gpu_monitor_test.cpp
    IndexFileAccessTest.cpp
    MathUtilsTest.cpp
    Minimap2IndexTest.cpp
    ModBaseEncoderTest.cpp
    ModelKitsTest.cpp
    ModelMetadataTest.cpp
    ModelUtilsTest.cpp
    MotifMatcherTest.cpp
    PairingNodeTest.cpp
    PipelineTest.cpp
    PolyACalculatorTest.cpp
    ReadFilterNodeTest.cpp
    ReadTest.cpp
    RealignMovesTest.cpp
    ResumeLoaderTest.cpp
    RNASplitTest.cpp
    SampleSheetTests.cpp
    SequenceUtilsTest.cpp
    StereoDuplexTest.cpp
    StitchTest.cpp
    StringUtilsTest.cpp
    TensorUtilsTest.cpp
    TimeUtilsTest.cpp
    TrimTest.cpp
)

if (NOT IOS)
    target_sources(dorado_tests
        PRIVATE
            # No FAST5 or POD5 on iOS
            Fast5DataLoaderTest.cpp
            Pod5DataLoaderTest.cpp
            # No dorado_io_lib on iOS
            ModelFinderTest.cpp
    )
endif()

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_tests PRIVATE MetalLinearTest.cpp)
    else()
        target_sources(dorado_tests PRIVATE cuda_utils_test.cpp)
    endif()
endif()


add_executable(dorado_smoke_tests
    main.cpp
    NodeSmokeTest.cpp
)


foreach(TEST_BIN dorado_tests dorado_smoke_tests)
    if (DORADO_ENABLE_PCH)
        target_precompile_headers(${TEST_BIN} REUSE_FROM dorado_lib)
    endif()

    target_link_libraries(${TEST_BIN} PRIVATE
        PRIVATE
            dorado_lib
            dorado_models_lib
            dorado_basecall
            dorado_modbase
            dorado_compat
            minimap2
            ${ZLIB_LIBRARIES}
    )
    if (NOT IOS)
        target_link_libraries(${TEST_BIN}
            PRIVATE
                dorado_io_lib
                ${POD5_LIBRARIES}
        )
    endif()

    target_include_directories(${TEST_BIN}
        SYSTEM
        PRIVATE
        ${DORADO_3RD_PARTY_SOURCE}/catch2
        )
    
    enable_warnings_as_errors(${TEST_BIN})

    add_test(
        NAME ${TEST_BIN}
        COMMAND ${TEST_BIN}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )

    if(NOT MSVC)
        # Add -Wno-trigraphs because one of the tests uses a qstring that has a trigraph sequence.
        target_compile_options(${TEST_BIN} PRIVATE "-Wno-trigraphs")
    endif()

    # The tests need to be able to find the libs in order to run.
    # We also want these libs to take priority over any installed on the system, so prepend them.
    if (MSVC)
        set_tests_properties(${TEST_BIN} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/bin")
        set_target_properties(${TEST_BIN} PROPERTIES LINK_OPTIONS "/ignore:4099")
    endif()

    # For some reason when TSan is enabled on x64/Linux we get a failure to load dependencies of torch, so add it explicitly to the path
    if (ECM_ENABLE_SANITIZERS AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
        set_tests_properties(${TEST_BIN} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${TORCH_LIB}/lib")
    endif()

    if (IOS)
        set_target_properties(${TEST_BIN} PROPERTIES
            BUNDLE True
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.ont.${TEST_BIN}"
            MACOSX_BUNDLE_BUNDLE_NAME ${TEST_BIN}
            MACOSX_BUNDLE_BUNDLE_VERSION "0.1"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
        )
    endif()
endforeach()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()
