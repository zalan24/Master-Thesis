include(external)

download_external(
    spirv_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    # GIT_TAG        1.5.4.raytracing.fixed
)

external(
    spirv_headers
    INSTALL ON
    TARGETS
        # SPIRV-Headers::SPIRV-Headers
    PROJECT_OPTIONS
        -DSPIRV_HEADERS_SKIP_EXAMPLES=ON
)
