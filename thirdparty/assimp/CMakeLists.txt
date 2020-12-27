include(FetchContent)

FetchContent_Declare(
    assimp_external
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG        v.5.0.0.rc1
)

FetchContent_GetProperties(assimp_external)
if(NOT assimp_external_POPULATED)
  FetchContent_Populate(assimp_external)
  set(ASSIMP_BUILD_TESTS OFF)
  add_subdirectory(${assimp_external_SOURCE_DIR} ${assimp_external_BINARY_DIR})
endif()

add_library(ASSIMP_LIB INTERFACE)
target_include_directories(ASSIMP_LIB SYSTEM INTERFACE "${assimp_external_BINARY_DIR}/include" "${assimp_external_SOURCE_DIR}/include")
target_link_libraries(ASSIMP_LIB INTERFACE assimp)

