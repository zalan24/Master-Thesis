file(GLOB RESOURCE_BINDINGS *.cpp *.h *.hpp)

add_library(AnimationsResourceBindings STATIC ${RESOURCE_BINDINGS})

set_target_properties(  AnimationsResourceBindings
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsResourceBindings PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsResourceBindings PUBLIC
    AnimationsFramegraph
)
