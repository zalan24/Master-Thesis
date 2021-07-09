file(GLOB PLAYGROUND_GAME_SOURCES *.cpp *.h *.hpp)

add_library(PlaygroundLib STATIC ${PLAYGROUND_GAME_SOURCES})

set_target_properties(  PlaygroundLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(PlaygroundLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(PlaygroundLib PUBLIC
    Game3D
    playgroundShaders
    stb_external
    cimg_external
)
