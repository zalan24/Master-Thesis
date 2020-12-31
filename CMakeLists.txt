cmake_minimum_required (VERSION 3.11.4)
project (Animations LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(C_STANDARD 11)

option(BUILD_TESTS "Build TESTS" ON)
set(SANITIZER "undefined" CACHE STRING "Sanitizer to use (address; undefined; leak)")
set_property(CACHE SANITIZER PROPERTY STRINGS address undefined leak)
if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    option(USE_SANITIZERS "Use sanitizers, only works on clang and gcc" ON)
else()
    option(USE_SANITIZERS "Use sanitizers, only works on clang and gcc" OFF)
endif()

set(SANITIZE_COMPILE_FLAGS "")
set(SANITIZE_LINK_FLAGS "")

if (USE_SANITIZERS)
    include(Sanitizer)
endif (USE_SANITIZERS)

option(FAST_MATH "Fast math" OFF)
if (FAST_MATH)
    include(FMath)
endif(FAST_MATH)

set(WARNINGFLAGS "")
set(TEST_WARNINGFLAGS "")
option(WERROR "Warnings are displayed as errors" OFF)
if (WERROR)
    include(Werror)
endif (WERROR)
include("${CMAKE_CXX_COMPILER_ID}")
set(TARGET_LINK_FLAGS "${SANITIZE_LINK_FLAGS}")
set(TARGETCOMPILEFLAGS "${WARNINGFLAGS} ${PLATFORM_FLAGS} ${SANITIZE_COMPILE_FLAGS}")
set(TEST_TARGETCOMPILEFLAGS "${TEST_WARNINGFLAGS} ${TEST_PLATFORM_FLAGS} ${SANITIZE_COMPILE_FLAGS}")

option(MICROPROFILE_ENABLED "Enabled microprofiler for drv" ON)
if (MICROPROFILE_ENABLED)
    add_definitions(-DMICROPROFILE_ENABLED)
endif()

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

# file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(thirdparty)
add_subdirectory(src)

if (BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif (BUILD_TESTS)