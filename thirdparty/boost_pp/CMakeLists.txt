include(external)

download_external(
    boost_pp
    GIT_REPOSITORY https://github.com/boostorg/preprocessor.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_pp_external INTERFACE)
target_include_directories(boost_pp_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_pp/src/include)
