include(external)

download_external(
    base64
    GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
)

add_library(base64_external INTERFACE)
target_include_directories(base64_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/base64/src)
