include(external)

download_external(
    cimg
    GIT_REPOSITORY https://github.com/dtschump/CImg.git
    GIT_TAG        v.2.9.7
)

add_library(cimg_external INTERFACE)
target_include_directories(cimg_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/cimg/src)
