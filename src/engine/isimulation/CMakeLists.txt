file(GLOB IRENDERER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsISimulation STATIC ${IRENDERER_SOURCES})

set_target_properties(  AnimationsISimulation
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsISimulation PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsISimulation PUBLIC
    AnimationsFramegraph
)
