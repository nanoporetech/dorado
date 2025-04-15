option(DORADO_DISABLE_TESTS "Disable building the tests" OFF)
option(DORADO_RUN_TESTS "Run the tests. Certain platforms don't have the infrastructure to do this atm" ON)
option(DORADO_ENABLE_BENCHMARK_TESTS "Enable benchmarks to run as part of the tests" OFF)

if (DORADO_DISABLE_TESTS)
    return()
endif()

# dorado_tests
add_executable(dorado_tests
    AdapterDetectorTest.cpp
    AlignerTest.cpp
    alignment_processing_items_test.cpp
    arg_parse_ext_test.cpp
    AsyncQueueTest.cpp
    async_task_executor_test.cpp
    BamReaderTest.cpp
    BamUtilsTest.cpp
    BamWriterTest.cpp
    BarcodeClassifierSelectorTest.cpp
    BarcodeClassifierTest.cpp
    BarcodeDemuxerNodeTest.cpp
    BasecallModelConfigTest.cpp
    BatchParamsTest.cpp
    bed_file_test.cpp
    CigarTest.cpp
    CliUtilsTest.cpp
    context_container_test.cpp
    CustomBarcodeParserTest.cpp
    DuplexReadTaggingNodeTest.cpp
    DuplexSplitTest.cpp
    fasta_reader_test.cpp
    fastq_reader_test.cpp
    Fast5DataLoaderTest.cpp
    FastxRandomReaderTest.cpp
    FileInfoTest.cpp
    gpu_monitor_test.cpp
    gzip_reader_test.cpp
    HtsFileTest.cpp
    IndexFileAccessTest.cpp
    MathUtilsTest.cpp
    MergeHeadersTest.cpp
    Minimap2IndexTest.cpp
    ModBaseChunkTest.cpp
    ModBaseConfigTest.cpp
    ModBaseEncoderTest.cpp
    ModelKitsTest.cpp
    ModelMetadataTest.cpp
    ModelSearchTest.cpp
    ModelUtilsTest.cpp
    MotifMatcherTest.cpp
    myers_test.cpp
    multi_queue_thread_pool_test.cpp
    PairingNodeTest.cpp
    PipelineTest.cpp
    Pod5DataLoaderTest.cpp
    PolyACalculatorTest.cpp
    PostConditionTest.cpp
    priority_task_queue_test.cpp
    ReadFilterNodeTest.cpp
    ReadForwarderNodeTest.cpp
    ReadTest.cpp
    RealignMovesTest.cpp
    ResumeLoaderTest.cpp
    RNASplitTest.cpp
    SampleSheetTests.cpp
    SamUtilsTest.cpp
    ScaledDotProductAttention.cpp
    SequenceUtilsTest.cpp
    StereoDuplexTest.cpp
    StitchTest.cpp
    StringUtilsTest.cpp
    synchronisation_test.cpp
    TensorUtilsTest.cpp
    TimeUtilsTest.cpp
    TrimTest.cpp
    PafUtilsTest.cpp
    SecondaryDecodeVariantsTest.cpp
    SecondaryNormalizeVariantsTest.cpp
    SecondaryMergeVCSamplesTest.cpp
    SecondarySampleTest.cpp
    SecondaryTrimTest.cpp
    SecondaryWindowTest.cpp
    RleTest.cpp
    CorrectionWindowTest.cpp
)

if (DORADO_GPU_BUILD)
    if(APPLE)
        target_sources(dorado_tests PRIVATE MetalLinearTest.cpp)
    else()
        target_sources(dorado_tests PRIVATE cuda_utils_test.cpp)
    endif()
endif()


# dorado_smoke_tests
add_executable(dorado_smoke_tests
    NodeSmokeTest.cpp
)


# dorado_tests_common
add_library(dorado_tests_common STATIC
    main.cpp
    TestUtils.cpp
)
enable_warnings_as_errors(dorado_tests_common)
target_link_libraries(dorado_tests_common
    PUBLIC
        dorado_torch_utils
        dorado_utils
        dorado_compat
        dorado_secondary
        Catch2::Catch2
)
target_compile_definitions(dorado_tests_common
    PUBLIC
        DORADO_ENABLE_BENCHMARK_TESTS=$<BOOL:${DORADO_ENABLE_BENCHMARK_TESTS}>
        # Workaround torch defining a CATCH_CHECK() macro that conflicts with catch2
        CATCH_CONFIG_PREFIX_ALL=1
)

# Finish setting up each target and add them as tests.
foreach(TEST_BIN dorado_tests dorado_smoke_tests)
    if (DORADO_ENABLE_PCH)
        target_precompile_headers(${TEST_BIN} REUSE_FROM dorado_lib)
    endif()

    target_link_libraries(${TEST_BIN}
        PRIVATE
            dorado_tests_common
            dorado_lib
            dorado_models_lib
            dorado_model_downloader_lib
            dorado_basecall
            dorado_modbase
            dorado_secondary
            minimap2
            ${ZLIB_LIBRARIES}
            dorado_io_lib
            dorado_file_info
            toml11::toml11
    )

    enable_warnings_as_errors(${TEST_BIN})
    if (MSVC)
        set_property(TARGET ${TEST_BIN} APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    else()
        # Add -Wno-trigraphs because one of the tests uses a qstring that has a trigraph sequence.
        target_compile_options(${TEST_BIN} PRIVATE "-Wno-trigraphs")
    endif()

    # Don't add the test if we can't run it
    if (NOT DORADO_RUN_TESTS)
        continue()
    endif()

    # Add the test.
    add_test(
        NAME ${TEST_BIN}
        COMMAND ${TEST_BIN}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    # The tests need to be able to find the libs in order to run.
    # We also want these libs to take priority over any installed on the system, so prepend them.
    if (MSVC)
        set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/bin")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_prepend:${CMAKE_INSTALL_PREFIX}/lib")
    endif()

    # For some reason when TSan is enabled on x64/Linux we get a failure to load dependencies of torch, so add it explicitly to the path
    if (ECM_ENABLE_SANITIZERS AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64") AND (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
        set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${TORCH_LIB}/lib")
    endif()

    # Enable the metal validation layer when running tests.
    if (APPLE)
        set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "MTL_DEBUG_LAYER=set:1")
        # We can't have warnings as errors because of the MPS backend in torch. Even enabling it in
        # logging mode is too spammy.
        #set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "MTL_DEBUG_LAYER_WARNING_MODE=set:nslog")

        # It would be nice to enable this fully for the simulator, however the simulator simulates old hardware
        # which we don't plan on supporting (256MB maxBufferLength, lack of compute memory barriers, etc...).
        #set_property(TEST ${TEST_BIN} APPEND PROPERTY ENVIRONMENT_MODIFICATION "SIMCTL_CHILD_MTL_DEBUG_LAYER=set:1")
    endif()
endforeach()

# GCC 8 ICEs trying to compile this file with ASAN+optimisations enabled, so knock down the optimisation to try and help it out.
if (ECM_ENABLE_SANITIZERS AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0))
    set_source_files_properties(TrimTest.cpp PROPERTIES COMPILE_OPTIONS "-O0")
endif()

# We don't build as PIC by default on Linux, so we fail to create a dynamic library there.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Create a dynamic library that checks that all symbols are available.
    # If this fails to link then we're not providing all of the libs for the symbols
    # that we use.
    add_library(dorado_symbol_test SHARED symbol_test.cpp)
    enable_warnings_as_errors(dorado_symbol_test)
    target_link_libraries(dorado_symbol_test PUBLIC dorado_lib dorado_models_lib)
    if(MSVC)
        set_property(TARGET dorado_symbol_test APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    endif()
endif()
