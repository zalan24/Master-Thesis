file(GLOB SHADER_COMPILER_SOURCES *.cpp *.h *.hpp)

add_executable(AnimationsShaderCompiler ${SHADER_COMPILER_SOURCES})

set_target_properties(  AnimationsShaderCompiler
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(AnimationsShaderCompiler PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsShaderCompiler PRIVATE cli11_external glslang_external spirv_tools_external simple_cpp_external hashlib4cpp_external)
target_link_libraries(AnimationsShaderCompiler PUBLIC
    AnimationsIO
    AnimationsShaderManager
    AnimationFeatures
    AnimationsSerializable
    RuntimeStats
)
