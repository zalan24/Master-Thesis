file(GLOB SHADER_COMPILER_SOURCES *.cpp *.h *.hpp)

add_library(ShaderCompilerLib STATIC ${SHADER_COMPILER_SOURCES})

set_target_properties(  ShaderCompilerLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderCompilerLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
# target_link_libraries(ShaderCompilerLib PRIVATE simple_cpp_external)
target_link_libraries(ShaderCompilerLib PUBLIC
    AnimationsIO
    AnimationsSerializable
    ShaderBin
    AnimationsTools
    ShaderPreprocessorLib
    # ShaderCompilerLib
    # ShaderManager
    # AnimationFeatures
    # RuntimeStats
)
