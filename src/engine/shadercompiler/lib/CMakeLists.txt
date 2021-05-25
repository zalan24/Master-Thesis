file(GLOB SHADER_COMPILER_LIB_SOURCES *.cpp *.h *.hpp)

add_library(ShaderCompilerLib ${SHADER_COMPILER_LIB_SOURCES})

set_target_properties(  ShaderCompilerLib
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderCompilerLib PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderCompilerLib PUBLIC
    DrvConfigs
    # ShaderBin
    # AnimationsIO
    # ShaderManager
    FeaturesLib
    SerializableLib
    # RuntimeStats
)
