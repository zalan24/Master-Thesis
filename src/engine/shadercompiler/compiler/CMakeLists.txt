file(GLOB SHADER_COMPILER_SOURCES *.cpp *.h *.hpp)

add_library(ShaderCompilerLib STATIC ${SHADER_COMPILER_SOURCES})

set_target_properties(  ShaderCompilerLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderCompilerLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderCompilerLib PRIVATE glslang_external spirv_tools_external)
target_link_libraries(ShaderCompilerLib PUBLIC
    AnimationsIO
    SerializableLib
    ShaderBin
    CoreTools
    ShaderPreprocessorLib
    Logging
    # ShaderCompilerLib
    # ShaderManager
    # FeaturesLib
    # RuntimeStats
)
