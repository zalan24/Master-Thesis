include(external)

download_external(
    boost_mpl
    GIT_REPOSITORY https://github.com/boostorg/mpl.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_mpl_external INTERFACE)
target_include_directories(boost_mpl_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_mpl/src/include)
target_link_libraries(boost_mpl_external INTERFACE boost_core_external boost_utility_external)
