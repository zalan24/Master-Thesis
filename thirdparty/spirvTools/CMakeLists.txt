include(FetchContent)

FetchContent_Declare(
    spriv_tools_external
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG        v2020.6
)

FetchContent_GetProperties(spriv_tools_external)
if(NOT spriv_tools_external_POPULATED)
  FetchContent_Populate(spriv_tools_external)
  prep_thirdparty(${spriv_tools_external_SOURCE_DIR} "${spriv_tools_external_SOURCE_DIR}_fixed")
  add_subdirectory("${spriv_tools_external_SOURCE_DIR}_fixed" ${spriv_tools_external_BINARY_DIR})
endif()
set(FIXED "${spriv_tools_external_SOURCE_DIR}_fixed")

add_library(SPIRV_TOOLS_LIB INTERFACE)
target_include_directories(SPIRV_TOOLS_LIB SYSTEM INTERFACE "${FIXED}/include")
target_link_libraries(SPIRV_TOOLS_LIB INTERFACE SPIRV_HEADERS_LIB)
