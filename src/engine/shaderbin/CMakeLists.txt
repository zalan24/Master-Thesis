file(GLOB SHADER_BIN_SOURCES *.cpp *.h *.hpp)

add_library(ShaderBin STATIC ${SHADER_BIN_SOURCES})

set_target_properties(  ShaderBin
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(ShaderBin PUBLIC . ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ShaderBin PUBLIC DrvCommon)