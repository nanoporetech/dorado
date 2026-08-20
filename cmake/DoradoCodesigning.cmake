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
    file(RELATIVE_PATH RELATIVE_PATH "${DORADO_SIGNING_ROOT}" "${EXECUTABLE}")

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

        execute_process(
            COMMAND
                codesign
                --verify
                --deep
                --strict
                --verbose=4
                "${EXECUTABLE}"
            RESULT_VARIABLE
                VERIFY_RESULT
        )
        if (NOT ${VERIFY_RESULT} EQUAL 0)
            message(FATAL_ERROR "Signature verification failed for ${EXECUTABLE}")
        endif()

        # `codesign --verify` only checks that the signature is internally
        # consistent.  Notarization additionally requires a Developer ID
        # Application certificate, a secure timestamp, and hardened runtime.
        # Check all three here so a wrongly configured CI identity fails before
        # uploading the archive to Apple.
        execute_process(
            COMMAND
                codesign
                --display
                --verbose=4
                "${EXECUTABLE}"
            RESULT_VARIABLE
                SIGNATURE_DETAILS_RESULT
            ERROR_VARIABLE
                SIGNATURE_DETAILS
        )
        if (NOT ${SIGNATURE_DETAILS_RESULT} EQUAL 0)
            message(FATAL_ERROR "Could not inspect the signature for ${EXECUTABLE}")
        endif()
        if (NOT "${SIGNATURE_DETAILS}" MATCHES "Authority=Developer ID Application:")
            message(FATAL_ERROR
                "${EXECUTABLE} was not signed with a Developer ID Application certificate. "
                "Set APPLE_CODESIGN_IDENTITY to a Developer ID Application identity.\n"
                "${SIGNATURE_DETAILS}"
            )
        endif()
        if (NOT "${SIGNATURE_DETAILS}" MATCHES "Timestamp=")
            message(FATAL_ERROR "${EXECUTABLE} is missing a secure signing timestamp.\n${SIGNATURE_DETAILS}")
        endif()
        if (NOT "${SIGNATURE_DETAILS}" MATCHES "Runtime Version=")
            message(FATAL_ERROR "${EXECUTABLE} does not have the hardened runtime enabled.\n${SIGNATURE_DETAILS}")
        endif()
        math(EXPR DORADO_SIGNED_COUNT "${DORADO_SIGNED_COUNT} + 1")
        set(DORADO_SIGNED_COUNT "${DORADO_SIGNED_COUNT}" PARENT_SCOPE)
    else()
        message(STATUS "Skipping ${RELATIVE_PATH} since it's not an executable")
    endif()
endfunction()

# CPack's pre-build scripts retain the project's CMAKE_INSTALL_PREFIX (usually
# /usr/local), so it must not be used to find the package payload. Component
# archives have one staging subtree per component, each with its own archive
# top-level directory. Search the complete CPack staging tree.
set(DORADO_SIGNING_ROOT "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
if ("${DORADO_SIGNING_ROOT}" STREQUAL "")
    message(FATAL_ERROR "CPACK_TEMPORARY_INSTALL_DIRECTORY is not set")
endif()
message(STATUS "Signing package payload in ${DORADO_SIGNING_ROOT}")

# Sign every Mach-O payload in every component, including dependencies nested
# in frameworks or subdirectories. The file type check in sign_executable()
# skips data files and symlinks.
file(GLOB_RECURSE SIGNING_CANDIDATES
    LIST_DIRECTORIES FALSE
    "${DORADO_SIGNING_ROOT}/*"
)
if ("${SIGNING_CANDIDATES}" STREQUAL "")
    message(FATAL_ERROR "No files were found to sign below ${DORADO_SIGNING_ROOT}")
endif()
set(DORADO_SIGNED_COUNT 0)
foreach(CANDIDATE ${SIGNING_CANDIDATES})
    sign_executable(${CANDIDATE})
endforeach()
if (DORADO_SIGNED_COUNT EQUAL 0)
    message(FATAL_ERROR "No Mach-O executables or libraries were signed below ${DORADO_SIGNING_ROOT}")
endif()
