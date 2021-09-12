file(GLOB SHADER_COMPILER_SOURCES *.cpp *.h *.hpp)

add_library(ShaderPreprocessorLib STATIC ${SHADER_COMPILER_SOURCES})

set_target_properties(  ShaderPreprocessorLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderPreprocessorLib PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderPreprocessorLib PRIVATE simple_cpp_external)
target_link_libraries(ShaderPreprocessorLib PUBLIC
    AnimationsIO
    SerializableLib
    ShaderBin
    ShaderManager
    CoreTools
    Logging
)
