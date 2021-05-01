file(GLOB IRENDERER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsIRenderer STATIC ${IRENDERER_SOURCES})

set_target_properties(  AnimationsIRenderer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsIRenderer PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsIRenderer PUBLIC
    EngineFramegraph
)
