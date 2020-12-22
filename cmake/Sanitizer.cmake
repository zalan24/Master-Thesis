if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    message("Sanitizer turned on")

    set (SANITIZE_COMPILE_FLAGS "-fsanitize=${SANITIZER} -fno-optimize-sibling-calls")
    set (SANITIZE_LINK_FLAGS "-fsanitize=${SANITIZER}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Sanitizer turned on")

    set (SANITIZE_COMPILE_FLAGS "-fsanitize=${SANITIZER}")
    set (SANITIZE_LINK_FLAGS "-fsanitize=${SANITIZER}")
else ()
    message("WARNING: sanitizers only work with clang and gcc")
endif ()