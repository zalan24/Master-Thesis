include(FetchContent)

FetchContent_Declare(
    imgui_external
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.71
)

FetchContent_GetProperties(imgui_external)
if(NOT imgui_external_POPULATED)
  FetchContent_Populate(imgui_external)
  prep_thirdparty(${imgui_external_SOURCE_DIR} "${imgui_external_SOURCE_DIR}_fixed")
  # add_subdirectory(${FIXED} ${imgui_external_BINARY_DIR})
endif()
set(FIXED "${imgui_external_SOURCE_DIR}_fixed")

# TODO vulkan
# TODO replace downloader

file(GLOB IMGUI_SOURCES ${FIXED}/*.cpp ${FIXED}/*.h ${FIXED}/*.hpp)
file(GLOB IMGUI_SOURCES_EXAMPLES ${FIXED}/examples/*.cpp ${FIXED}/examples/*.h ${FIXED}/examples/*.hpp)
set(IMGUI_SOURCES_EXAMPLES
  ${FIXED}/examples/imgui_impl_glfw.h
  ${FIXED}/examples/imgui_impl_glfw.cpp
  ${FIXED}/examples/imgui_impl_opengl3.h
  ${FIXED}/examples/imgui_impl_opengl3.cpp
  ${FIXED}/examples/imgui_impl_opengl2.h
  ${FIXED}/examples/imgui_impl_opengl2.cpp
)
add_library(IMGUI_LIB STATIC ${IMGUI_SOURCES} ${IMGUI_SOURCES_EXAMPLES})
target_include_directories(IMGUI_LIB SYSTEM PUBLIC "${imgui_external_BINARY_DIR}" "${FIXED}")
target_include_directories(IMGUI_LIB SYSTEM PRIVATE "${imgui_external_BINARY_DIR}/examples" "${FIXED}/examples")
# target_link_libraries(IMGUI_LIB PRIVATE glfw_external GLAD_LIB)
# target_compile_definitions(IMGUI_LIB PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)