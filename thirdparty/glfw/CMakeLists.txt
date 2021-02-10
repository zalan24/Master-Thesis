include(external)

download_external(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.2
)

external(
    glfw
    INSTALL OFF
    TARGETS
        glfw
    PROJECT_OPTIONS
        -DBUILD_SHARED_LIBS=OFF
        -DGLFW_BUILD_EXAMPLES=OFF
        -DGLFW_BUILD_TESTS=OFF
        -DGLFW_BUILD_DOCS=OFF
        -DGLFW_VULKAN_STATIC=ON
)
