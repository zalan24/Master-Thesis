file(GLOB SHADER_MANAGER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsShaderManager STATIC ${SHADER_MANAGER_SOURCES})

set_target_properties(  AnimationsShaderManager
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsShaderManager PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsShaderManager PRIVATE AnimationsSerializable)