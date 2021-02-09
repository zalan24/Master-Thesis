include(external)

download_external(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
)

add_library(stb_external INTERFACE)
target_include_directories(stb_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/stb/src)
