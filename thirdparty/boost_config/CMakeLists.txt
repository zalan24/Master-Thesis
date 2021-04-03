include(external)

download_external(
    boost_config
    GIT_REPOSITORY https://github.com/boostorg/config.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_config_external INTERFACE)
target_include_directories(boost_config_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_config/src/include)
