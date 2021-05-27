include(external)

download_external(
    boost_static_assert
    GIT_REPOSITORY https://github.com/boostorg/static_assert.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_static_assert_external INTERFACE)
target_include_directories(boost_static_assert_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_static_assert/src/include)
