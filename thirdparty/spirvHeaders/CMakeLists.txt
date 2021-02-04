include(FetchContent)

FetchContent_Declare(
    spriv_headers_external
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG        1.5.4
)

FetchContent_GetProperties(spriv_headers_external)
if(NOT spriv_headers_external_POPULATED)
  FetchContent_Populate(spriv_headers_external)
  prep_thirdparty(${spriv_headers_external_SOURCE_DIR} "${spriv_headers_external_SOURCE_DIR}_fixed")
  add_subdirectory("${spriv_headers_external_SOURCE_DIR}_fixed" ${spriv_headers_external_BINARY_DIR})
endif()
set(FIXED "${spriv_headers_external_SOURCE_DIR}_fixed")

add_library(SPIRV_HEADERS_LIB INTERFACE)
target_include_directories(SPIRV_HEADERS_LIB SYSTEM INTERFACE "${FIXED}/include")
# target_link_libraries(SPIRV_TOOLS_LIB INTERFACE )
