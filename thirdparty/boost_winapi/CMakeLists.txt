include(external)

download_external(
    boost_winapi
    GIT_REPOSITORY https://github.com/boostorg/winapi.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_winapi_external INTERFACE)
target_include_directories(boost_winapi_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_winapi/src/include)
target_link_libraries(boost_winapi_external INTERFACE boost_predef_external)
