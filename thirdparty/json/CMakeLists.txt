include(external)

download_external(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.9.1
)

add_library(json_external INTERFACE)
target_include_directories(json_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/json/src/include)
