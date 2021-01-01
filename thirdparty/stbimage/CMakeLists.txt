include(FetchContent)

FetchContent_Declare(
    stb_external
    GIT_REPOSITORY https://github.com/nothings/stb.git
)

FetchContent_GetProperties(stb_external)
if(NOT stb_external_POPULATED)
  FetchContent_Populate(stb_external)
  prep_thirdparty(${stb_external_SOURCE_DIR} "${stb_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${stb_external_SOURCE_DIR}_fixed")

add_library(STB_LIB INTERFACE)
add_dependencies(STB_LIB stb_external)
target_include_directories(STB_LIB SYSTEM INTERFACE ${FIXED})
