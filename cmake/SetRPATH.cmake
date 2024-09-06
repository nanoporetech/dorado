if (NOT DORADO_DISABLE_DORADO AND NOT WIN32)
    # Set up RPATHs so we can find dependencies
    set(CMAKE_SKIP_RPATH FALSE)
    # Note: we don't need the relative lib dir if everything is in
    if (APPLE)
        set(CMAKE_INSTALL_RPATH "@executable_path/;@executable_path/../lib")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib/;$ORIGIN")
    endif()
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
endif()
