include(external)

download_external(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
	GIT_TAG        11.1.0
)

external(
    glslang
    INSTALL ON
    TARGETS
        OSDependent
        OGLCompiler
        glslang
        SPIRV
)
