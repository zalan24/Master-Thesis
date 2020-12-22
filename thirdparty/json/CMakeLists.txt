include(FetchContent)

FetchContent_Declare(
    json_external
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.7.0
)

FetchContent_GetProperties(json_external)
if(NOT json_externalPOPULATED)
  FetchContent_Populate(json_external)
endif()
add_library(JSON_LIB INTERFACE)
add_dependencies(JSON_LIB json_external)
target_include_directories(JSON_LIB SYSTEM INTERFACE ${json_external_SOURCE_DIR}/include)
