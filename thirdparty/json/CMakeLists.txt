include(FetchContent)

FetchContent_Declare(
    json_external
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.9.1
)

FetchContent_GetProperties(json_external)
if(NOT json_externalPOPULATED)
  FetchContent_Populate(json_external)
  prep_thirdparty(${json_external_SOURCE_DIR} "${json_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${json_external_SOURCE_DIR}_fixed")

add_library(JSON_LIB INTERFACE)
add_dependencies(JSON_LIB json_external)
target_include_directories(JSON_LIB SYSTEM INTERFACE ${FIXED}/include)
