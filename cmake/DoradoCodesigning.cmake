# Helper script to codesign executables as part of CPack packaging.
#
# Note that this script is intended to run after CPack install phase but before the build.
#

# x64 builds can't be notarized on our CI machines and codesigning by itself doesn't
# affect the end user, but if we enable the hardened runtime then we get crashes when
# basecalling due to torch jitting code so skip signing entirely.
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(STATUS "Skipping code signing on x64 due to torch jitting")
    return()
endif()

# Grab the identity from the environment
set(DORADO_CODESIGN_IDENTITY "$ENV{APPLE_CODESIGN_IDENTITY}")
if ("${DORADO_CODESIGN_IDENTITY}" STREQUAL "")
    message(FATAL_ERROR "Trying to sign a build without setting a codesign identity (APPLE_CODESIGN_IDENTITY envvar)")
endif()

function(sign_executable EXECUTABLE)
    file(RELATIVE_PATH RELATIVE_PATH "${CPACK_TEMPORARY_INSTALL_DIRECTORY}" "${EXECUTABLE}")

    # Check that this is an executable and not data or a symlink
    execute_process(
        COMMAND
            file
            --no-dereference
            "${EXECUTABLE}"
        RESULT_VARIABLE
            FILE_RESULT
        OUTPUT_VARIABLE
            FILE_OUTPUT
    )
    if (NOT ${FILE_RESULT} EQUAL 0)
        message(FATAL_ERROR "Failed to check whether ${EXECUTABLE} is an executable")
    endif()
    if ("${FILE_OUTPUT}" MATCHES " 64-bit executable " OR "${FILE_OUTPUT}" MATCHES " shared library ")
        message(STATUS "Signing ${RELATIVE_PATH}")
        execute_process(
            COMMAND
                codesign
                --sign "${DORADO_CODESIGN_IDENTITY}"
                --timestamp
                --options=runtime
                -vvvv
                "${EXECUTABLE}"
            RESULT_VARIABLE
                SIGNING_RESULT
        )
        if (NOT ${SIGNING_RESULT} EQUAL 0)
            message(FATAL_ERROR "Signing failed for ${EXECUTABLE}")
        endif()
    else()
        message(STATUS "Skipping ${RELATIVE_PATH} since it's not an executable")
    endif()
endfunction()

# Sign all the executables and libs we can find
file(GLOB EXECUTABLES "${CPACK_TEMPORARY_INSTALL_DIRECTORY}/bin/*")
foreach(EXECUTABLE ${EXECUTABLES})
    sign_executable(${EXECUTABLE})
endforeach()
file(GLOB LIBS "${CPACK_TEMPORARY_INSTALL_DIRECTORY}/lib/*")
foreach(LIB ${LIBS})
    sign_executable(${LIB})
endforeach()
