file(GLOB SHADER_MANAGER_SOURCES *.cpp *.h *.hpp)

add_library(AnimationsShaderManager ${SHADER_MANAGER_SOURCES})
add_dependencies(AnimationsShaderManager ShaderCodes)

set_target_properties(  AnimationsShaderManager
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS})
target_include_directories(AnimationsShaderManager PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationsShaderManager PRIVATE GLAD_LIB GLM_LIB MAPBOX_LIB)
