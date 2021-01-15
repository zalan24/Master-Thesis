file(GLOB RENDERER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsRenderer ${RENDERER_SOURCES})

set_target_properties(  AnimationsRenderer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsRenderer PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsRenderer PRIVATE GLAD_LIB GLM_LIB)
target_link_libraries(AnimationsRenderer PUBLIC
    AnimationsShaderManager
    AnimationsEntities
    AnimationsData
    AnimationsBuffer
    AnimationsGame
    AnimationsResources
)
