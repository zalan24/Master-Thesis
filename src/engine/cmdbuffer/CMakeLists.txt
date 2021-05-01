file(GLOB RENDER_PASS_SOURCES *.cpp *.h *.hpp)

add_library(EngineCmdBuffer STATIC ${RENDER_PASS_SOURCES})

set_target_properties(  EngineCmdBuffer
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(EngineCmdBuffer PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(EngineCmdBuffer PUBLIC
    AnimationsDrv
)
