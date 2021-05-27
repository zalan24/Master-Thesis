include(external)

download_external(
    boost_bind
    GIT_REPOSITORY https://github.com/boostorg/bind.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_bind_external INTERFACE)
target_include_directories(boost_bind_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_bind/src/include)
