include(FetchContent)

FetchContent_Declare(
    cli11_external
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v1.8.0
)

FetchContent_GetProperties(cli11_external)
if(NOT cli11_externalPOPULATED)
  FetchContent_Populate(cli11_external)
  prep_thirdparty(${cli11_external_SOURCE_DIR} "${cli11_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${cli11_external_SOURCE_DIR}_fixed")

add_library(CLI11_LIB INTERFACE)
add_dependencies(CLI11_LIB cli11_external)
target_include_directories(CLI11_LIB SYSTEM INTERFACE ${FIXED}/include)
