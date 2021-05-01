file(GLOB SHADER_MANAGER_SOURCES *.cpp *.h *.hpp)

add_library(ShaderManager STATIC ${SHADER_MANAGER_SOURCES})

set_target_properties(  ShaderManager
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderManager PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderManager PRIVATE AnimationsSerializable EngineFramegraph)
target_link_libraries(ShaderManager PUBLIC ShaderBin AnimationsDrv glm_external)