include(external)

download_external(
    boost_core
    GIT_REPOSITORY https://github.com/boostorg/core.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_core_external INTERFACE)
target_include_directories(boost_core_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_core/src/include)
