option(DORADO_DISABLE_TESTS "Disable building the tests" OFF)
option(DORADO_RUN_TESTS "Run the tests. Certain platforms don't have the infrastructure to do this atm" ON)

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
    BasecallerParamsTest.cpp
    bed_file_test.cpp
    CigarTest.cpp
    CliUtilsTest.cpp
    context_container_test.cpp
    CRFModelConfigTest.cpp
    CustomBarcodeParserTest.cpp
    DuplexReadTaggingNodeTest.cpp
    DuplexSplitTest.cpp
    fastq_reader_test.cpp
    FastxRandomReaderTest.cpp
    gpu_monitor_test.cpp
    HtsFileTest.cpp
    IndexFileAccessTest.cpp
    MathUtilsTest.cpp
    MergeHeadersTest.cpp
    Minimap2IndexTest.cpp
    ModBaseConfigTest.cpp
    ModBaseEncoderTest.cpp
    ModelKitsTest.cpp
    ModelMetadataTest.cpp
    ModelUtilsTest.cpp
    MotifMatcherTest.cpp
    myers_test.cpp
    multi_queue_thread_pool_test.cpp
    PairingNodeTest.cpp
    PipelineTest.cpp
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
    TrimRapidAdapterTest.cpp
    TrimTest.cpp
    PafUtilsTest.cpp
)
if (NOT IOS)
    target_sources(dorado_tests
        PRIVATE
            # No FAST5 or POD5 on iOS
            Fast5DataLoaderTest.cpp
            Pod5DataLoaderTest.cpp
            # No dorado_io_lib on iOS
            ModelSearchTest.cpp
    )
endif()
target_compile_definitions(dorado_tests
    PRIVATE
        DORADO_IOS_BUILD=$<BOOL:${IOS}>
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
target_link_libraries(dorado_tests_common
    PUBLIC
        dorado_torch_utils
        dorado_utils
        dorado_compat
)
target_include_directories(dorado_tests_common
    SYSTEM
    PUBLIC
        ${DORADO_3RD_PARTY_SOURCE}/catch2
)


# Setup/teardown for iOS tests
if (IOS AND DORADO_RUN_TESTS)
    set(IOS_TEST_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/run_ios_test.py)
    # I'm assuming that these tests will only be run in the sim on CI, so they
    # depend on the sim being launched.
    # Note: The test targets can still be built, but you'll need to install+launch
    # them manually on a real device.
    add_test(
        NAME launch_simulator
        COMMAND ${IOS_TEST_SCRIPT} --setup
    )
    set_tests_properties(launch_simulator
        PROPERTIES
            FIXTURES_SETUP interacting_with_device
    )
    # Kill the sim after so that it's not just hanging around.
    add_test(
        NAME kill_simulator_after
        COMMAND ${IOS_TEST_SCRIPT} --teardown
    )
    set_tests_properties(kill_simulator_after
        PROPERTIES
            FIXTURES_CLEANUP interacting_with_device
    )
endif()


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

    enable_warnings_as_errors(${TEST_BIN})
    if (MSVC)
        set_property(TARGET ${TEST_BIN} APPEND PROPERTY LINK_OPTIONS "/ignore:4099")
    else()
        # Add -Wno-trigraphs because one of the tests uses a qstring that has a trigraph sequence.
        target_compile_options(${TEST_BIN} PRIVATE "-Wno-trigraphs")
    endif()

    if (IOS)
        # default.metallib is generated, but we need it as a source file to package it,
        # so create it if it doesn't exist.
        set(METALLIB_PATH ${CMAKE_BINARY_DIR}/lib/default.metallib)
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
        file(TOUCH ${METALLIB_PATH})

        # Grab all the data files that we want to package.
        set(DATA_RESOURCES "${CMAKE_CURRENT_SOURCE_DIR}/data;${METALLIB_PATH}")

        # To package data into the app we need to make it a source file.
        target_sources(${TEST_BIN} PRIVATE ${DATA_RESOURCES})

        # iOS bundle names can't contain underscores.
        string(REPLACE "_" "-" IOS_BUNDLE_NAME "${TEST_BIN}")

        set_target_properties(${TEST_BIN} PROPERTIES
            BUNDLE TRUE
            MACOSX_BUNDLE_GUI_IDENTIFIER "com.ont.${IOS_BUNDLE_NAME}"
            MACOSX_BUNDLE_BUNDLE_NAME ${IOS_BUNDLE_NAME}
            MACOSX_BUNDLE_BUNDLE_VERSION "0.1"
            MACOSX_BUNDLE_SHORT_VERSION_STRING "0.1"
            RESOURCE "${DATA_RESOURCES}"
        )
    endif()

    # Don't add the test if we can't run it
    if (NOT DORADO_RUN_TESTS)
        continue()
    endif()

    # Add the test.
    if (IOS)
        add_test(
            NAME ${TEST_BIN}
            COMMAND
                ${IOS_TEST_SCRIPT}
                    --run
                    # This should be $<TARGET_FILE_DIR:${TEST_BIN}> but that's broken
                    # when building for the sim.
                    --app_path ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-iphonesimulator/${TEST_BIN}.app
                    --identifier com.ont.${IOS_BUNDLE_NAME}
                    --device booted
        )
        set_tests_properties(${TEST_BIN}
            PROPERTIES
                FIXTURES_REQUIRED interacting_with_device
        )
    else()
        add_test(
            NAME ${TEST_BIN}
            COMMAND ${TEST_BIN}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()

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
    target_link_libraries(dorado_symbol_test PUBLIC dorado_lib dorado_models_lib)
endif()
