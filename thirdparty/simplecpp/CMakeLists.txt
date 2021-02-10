include(external)

download_external(
    simple_cpp
    GIT_REPOSITORY https://github.com/danmar/simplecpp.git
)

add_library(simple_cpp_external STATIC ${PROJECT_BINARY_DIR}/3rdParty/simple_cpp/src/simplecpp.cpp)
set_target_properties(  simple_cpp_external
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(simple_cpp_external SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/3rdParty/simple_cpp/src)
