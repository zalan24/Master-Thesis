file(GLOB GAME_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsGame ${GAME_SOURCES})

set_target_properties(  AnimationsGame
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsGame PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsGame PUBLIC AnimationsEntities AnimationsIO)
