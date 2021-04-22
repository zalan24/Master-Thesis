include(external)

download_external(
    spirv_tools
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG        v2021.1
)

external(
  spirv_tools
    INSTALL ON
    CUSTOM_PRE_CMAKE "precmake.cmake"
    PROJECT_OPTIONS
        -DSPIRV_TOOLS_BUILD_STATIC=OFF
        -DSPIRV_SKIP_TESTS=ON
        "-DSPIRV-Headers_SOURCE_DIR:FILEPATH=${PROJECT_BINARY_DIR}/3rdParty/spirv_headers/src"
)

target_link_libraries(spirv_tools_external INTERFACE spirv_headers_external spirv_cross_external)
