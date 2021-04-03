include(external)

download_external(
    boost_align
    GIT_REPOSITORY https://github.com/boostorg/align.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_align_external INTERFACE)
target_include_directories(boost_align_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_align/src/include)
target_link_libraries(boost_align_external INTERFACE boost_config_external boost_assert_external)
