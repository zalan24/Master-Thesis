include(FetchContent)

FetchContent_Declare(
    stb_external
    GIT_REPOSITORY https://github.com/nothings/stb.git
)

FetchContent_GetProperties(stb_external)
if(NOT stb_external_POPULATED)
  FetchContent_Populate(stb_external)
endif()
add_library(STB_LIB INTERFACE)
add_dependencies(STB_LIB stb_external)
target_include_directories(STB_LIB SYSTEM INTERFACE ${stb_external_SOURCE_DIR})
