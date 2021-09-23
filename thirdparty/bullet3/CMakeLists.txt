include(external)

download_external(
    bullet3
    GIT_REPOSITORY https://github.com/bulletphysics/bullet3.git
    GIT_TAG        3.17
)

external(
  bullet3
    INSTALL ON
    # TARGETS
    # PROJECT_OPTIONS
)

# target_link_libraries(bullet3_external INTERFACE spirv_headers_external)