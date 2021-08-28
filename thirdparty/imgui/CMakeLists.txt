include(external)

download_external(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.84.2
)

set(IMGUI_DIR ${PROJECT_BINARY_DIR}/3rdParty/imgui/src)

file(GLOB IMGUI_SOURCES ${IMGUI_DIR}/*.cpp ${IMGUI_DIR}/*.h ${IMGUI_DIR}/*.hpp)
file(GLOB IMGUI_SOURCES_BACKEND ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)

add_library(ImGuiLib STATIC ${IMGUI_SOURCES} ${IMGUI_SOURCES_BACKEND})
target_include_directories(ImGuiLib SYSTEM PUBLIC ${IMGUI_DIR})
target_include_directories(ImGuiLib SYSTEM PRIVATE ${IMGUI_DIR}/backends)
target_link_libraries(ImGuiLib PUBLIC glfw_external vulkan_headers_external)

