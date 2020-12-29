file(GLOB ENGINE_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsEngine ${ENGINE_SOURCES})

set_target_properties(  AnimationsEngine
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsEngine PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsEngine PRIVATE
    JSON_LIB
)
target_link_libraries(AnimationsEngine PUBLIC
    AnimationsEntities
    AnimationsSerializable
    AnimationsIO
    AnimationsGame
    AnimationsRenderer
    AnimationsWindow
)
