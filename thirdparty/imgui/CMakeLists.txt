include(FetchContent)

FetchContent_Declare(
    imgui_external
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.71
)

FetchContent_GetProperties(imgui_external)
if(NOT imgui_external_POPULATED)
  FetchContent_Populate(imgui_external)
  # add_subdirectory(${imgui_external_SOURCE_DIR} ${imgui_external_BINARY_DIR})
endif()

file(GLOB IMGUI_SOURCES ${imgui_external_SOURCE_DIR}/*.cpp ${imgui_external_SOURCE_DIR}/*.h ${imgui_external_SOURCE_DIR}/*.hpp)
file(GLOB IMGUI_SOURCES_EXAMPLES ${imgui_external_SOURCE_DIR}/examples/*.cpp ${imgui_external_SOURCE_DIR}/examples/*.h ${imgui_external_SOURCE_DIR}/examples/*.hpp)
set(IMGUI_SOURCES_EXAMPLES
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_glfw.h
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_glfw.cpp
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_opengl3.h
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_opengl3.cpp
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_opengl2.h
  ${imgui_external_SOURCE_DIR}/examples/imgui_impl_opengl2.cpp
  )
add_library(IMGUI_LIB ${IMGUI_SOURCES} ${IMGUI_SOURCES_EXAMPLES})
target_include_directories(IMGUI_LIB SYSTEM PUBLIC "${imgui_external_BINARY_DIR}" "${imgui_external_SOURCE_DIR}")
target_include_directories(IMGUI_LIB SYSTEM PRIVATE "${imgui_external_BINARY_DIR}/examples" "${imgui_external_SOURCE_DIR}/examples")
target_link_libraries(IMGUI_LIB PRIVATE GLFW_LIB GLAD_LIB)
target_compile_definitions(IMGUI_LIB PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)