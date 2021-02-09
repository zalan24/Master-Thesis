include(external)

download_external(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v1.8.0
)

add_library(cli11_external INTERFACE)
target_include_directories(cli11_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/cli11/src/include)
