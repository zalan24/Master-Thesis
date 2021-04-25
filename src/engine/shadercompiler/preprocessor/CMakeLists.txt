file(GLOB SHADER_COMPILER_SOURCES *.cpp *.h *.hpp)

add_executable(ShaderPreprocessor ${SHADER_COMPILER_SOURCES})

set_target_properties(  ShaderPreprocessor
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderPreprocessor PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderPreprocessor PRIVATE cli11_external hashlib4cpp_external simple_cpp_external)
target_link_libraries(ShaderPreprocessor PUBLIC
    AnimationsIO
    AnimationsSerializable
    ShaderBin
    AnimationsTools
    # ShaderCompilerLib
    # ShaderManager
    # AnimationFeatures
    # RuntimeStats
)
