include(external)

download_external(
    boost_stacktrace
    GIT_REPOSITORY https://github.com/boostorg/stacktrace
    GIT_TAG        boost-1.75.0
)

add_library(boost_stacktrace_external INTERFACE)
target_include_directories(boost_stacktrace_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_stacktrace/src/include)
target_link_libraries(boost_stacktrace_external INTERFACE boost_config_external boost_core_external boost_winapi_external)
