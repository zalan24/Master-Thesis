include(external)

download_external(
    boost_utility
    GIT_REPOSITORY https://github.com/boostorg/utility.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_utility_external INTERFACE)
target_include_directories(boost_utility_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_utility/src/include)
