include(FetchContent)

FetchContent_Declare(
    test_external
    GIT_REPOSITORY https://github.com/drleq/CppUnitTestFramework.git
    GIT_TAG        1.1.0
)

FetchContent_GetProperties(test_external)
if(NOT test_external_POPULATED)
  FetchContent_Populate(test_external)
  prep_thirdparty(${test_external_SOURCE_DIR} "${test_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${test_external_SOURCE_DIR}_fixed")

add_library(TEST_LIB INTERFACE)
add_dependencies(TEST_LIB test_external)
target_include_directories(TEST_LIB SYSTEM INTERFACE ${FIXED})
