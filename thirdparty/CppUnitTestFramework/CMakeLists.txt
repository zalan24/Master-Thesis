include(external)

download_external(
    cpp_unit_test
    GIT_REPOSITORY https://github.com/drleq/CppUnitTestFramework.git
    GIT_TAG        1.1.0
)

add_library(cpp_unit_test_external INTERFACE)
target_include_directories(cpp_unit_test_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/cpp_unit_test/src)
