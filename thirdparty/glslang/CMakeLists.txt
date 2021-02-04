include(FetchContent)

FetchContent_Declare(
    glslang_external
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG        11.1.0
)

FetchContent_GetProperties(glslang_external)
if(NOT glslang_external_POPULATED)
  FetchContent_Populate(glslang_external)
  prep_thirdparty(${glslang_external_SOURCE_DIR} "${glslang_external_SOURCE_DIR}_fixed")
  add_subdirectory("${glslang_external_SOURCE_DIR}_fixed" ${glslang_external_BINARY_DIR})
endif()
set(FIXED "${glslang_external_SOURCE_DIR}_fixed")

add_library(Glslang_LIB INTERFACE)
target_include_directories(Glslang_LIB SYSTEM INTERFACE "${FIXED}")
target_link_libraries(Glslang_LIB INTERFACE glslang SPIRV)
