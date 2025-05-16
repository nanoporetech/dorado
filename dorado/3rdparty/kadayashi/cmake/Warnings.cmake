function(enable_warnings_as_errors TARGET_NAME)
    if(WIN32)
        # W4 - warning level 4.
        # WX - warnings as errors.
        # external:anglebrackets - treat <header> includes as external.
        # external:W0 - disable warnings for external code.
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W4 /WX /external:anglebrackets /external:W0>
            # Make MSVC conform to the C++ spec.
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/permissive->
            # toml11 sets this as PUBLIC and hence we get "command-line option '/Zc:preprocessor'
            # inconsistent with precompiled header" errors when using a PCH if we don't too.
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/Zc:preprocessor>
            # spdlog's bundled libfmt requires this, and we get PCH errors if we don't set it too.
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
        )
        target_compile_definitions(${TARGET_NAME} PRIVATE
            _CRT_SECURE_NO_WARNINGS
        )
    elseif(CMAKE_COMPILER_IS_GNUCXX)
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -Werror -Wundef -Wshadow
            -Wmissing-declarations
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -Werror -Wundef -Wshadow-all
            -Wmissing-prototypes
        )
    else()
        message(FATAL_ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()

function(disable_warnings TARGET_NAME)
    if(WIN32)
        target_compile_options(${TARGET_NAME} PRIVATE
            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W0>
        )
    endif()
endfunction()

function(check_linked_libs TARGET_NAME)
    if (ECM_ENABLE_SANITIZERS)
        # We don't ship these, so no need to check them.
        return()
    endif()

    if (APPLE)
        add_custom_command(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND echo "Checking linked libs..."
            # We shouldn't be linking to anything from homebrew.
            COMMAND sh -c "otool -L $<TARGET_FILE:${TARGET_NAME}> | grep -i /opt/homebrew ; test $? -eq 1"
            COMMAND sh -c "otool -L $<TARGET_FILE:${TARGET_NAME}> | grep -i /usr/local/opt ; test $? -eq 1"
            VERBATIM
        )
    endif()
endfunction()
