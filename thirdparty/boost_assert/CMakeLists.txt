include(external)

download_external(
    boost_assert
    GIT_REPOSITORY https://github.com/boostorg/assert.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_assert_external INTERFACE)
target_include_directories(boost_assert_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_assert/src/include)
