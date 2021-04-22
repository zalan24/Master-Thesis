include(external)

download_external(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
	GIT_TAG        11.2.0
)

external(
    glslang
    INSTALL ON
    CUSTOM_PRE_CMAKE "precmake.cmake"
    TARGETS
        OSDependent
        OGLCompiler
        glslang
        SPIRV
    PROJECT_OPTIONS
        -DBUILD_EXTERNAL=OFF
)
