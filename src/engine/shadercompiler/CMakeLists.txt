add_subdirectory(compiler)
add_subdirectory(preprocessor)

add_executable(ShaderPreprocessor shaderpreprocessor.cpp)

set_target_properties(  ShaderPreprocessor
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderPreprocessor PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderPreprocessor PRIVATE cli11_external)
target_link_libraries(ShaderPreprocessor PUBLIC ShaderPreprocessorLib)


add_executable(ShaderCompiler shadercompiler.cpp)

set_target_properties(  ShaderCompiler
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderCompiler PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderCompiler PRIVATE cli11_external)
target_link_libraries(ShaderCompiler PUBLIC ShaderCompilerLib)
