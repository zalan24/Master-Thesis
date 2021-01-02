include(FetchContent)

FetchContent_Declare(
    assimp_external
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG        v5.0.1
)

if ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
  # Workaround for assimp's terrible build system
  add_compile_options(-Wno-unused-command-line-argument -Wno-c99-extensions)
endif()

FetchContent_GetProperties(assimp_external)
if(NOT assimp_external_POPULATED)
  FetchContent_Populate(assimp_external)
  set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  prep_thirdparty(${assimp_external_SOURCE_DIR} "${assimp_external_SOURCE_DIR}_fixed")
  add_subdirectory("${assimp_external_SOURCE_DIR}_fixed" ${assimp_external_BINARY_DIR})
endif()
set(FIXED "${assimp_external_SOURCE_DIR}_fixed")

add_library(ASSIMP_LIB INTERFACE)
target_include_directories(ASSIMP_LIB SYSTEM INTERFACE "${assimp_external_BINARY_DIR}/include" "${FIXED}/include")
target_link_libraries(ASSIMP_LIB INTERFACE assimp)
