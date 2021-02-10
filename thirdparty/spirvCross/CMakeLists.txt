# include(FetchContent)

# FetchContent_Declare(
#     spirv_cross_external
#     GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
#     GIT_TAG        2021-01-15
# )

# FetchContent_GetProperties(spirv_cross_external)
# if(NOT spirv_cross_external_POPULATED)
#   FetchContent_Populate(spirv_cross_external)
#   prep_thirdparty(${spirv_cross_external_SOURCE_DIR} "${spirv_cross_external_SOURCE_DIR}_fixed")
#   add_subdirectory("${spirv_cross_external_SOURCE_DIR}_fixed" ${spirv_cross_external_BINARY_DIR})
# endif()
# set(FIXED "${spirv_cross_external_SOURCE_DIR}_fixed")

# add_library(Spirv_LIB INTERFACE)
# target_include_directories(Spirv_LIB SYSTEM INTERFACE "${FIXED}")
# # target_link_libraries(Spirv_LIB INTERFACE assimp)

include(external)

download_external(
    spirv_cross
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
    GIT_TAG        2021-01-15
)

external(
  spirv_cross
    INSTALL ON
    # TARGETS
    # PROJECT_OPTIONS
)

# target_link_libraries(spirv_cross_external INTERFACE spirv_headers_external)