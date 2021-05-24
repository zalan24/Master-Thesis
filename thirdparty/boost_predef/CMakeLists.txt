include(external)

download_external(
    boost_predef
    GIT_REPOSITORY https://github.com/boostorg/predef.git
    GIT_TAG        boost-1.75.0
)

external(
    boost_predef
    INSTALL OFF
    TARGETS Boost::predef
    PROJECT_OPTIONS
)


# add_library(boost_predef_external INTERFACE)
# target_include_directories(boost_predef_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_predef/src/include)
