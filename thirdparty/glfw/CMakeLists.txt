include(FetchContent)

FetchContent_Declare(
    glfw_external
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.2
)

FetchContent_GetProperties(glfw_external)
if(NOT glfw_external_POPULATED)
  FetchContent_Populate(glfw_external)
  prep_thirdparty(${glfw_external_SOURCE_DIR} "${glfw_external_SOURCE_DIR}_fixed")
  add_subdirectory("${glfw_external_SOURCE_DIR}_fixed" ${glfw_external_BINARY_DIR})
endif()
set(FIXED "${glfw_external_SOURCE_DIR}_fixed")

add_library(GLFW_LIB INTERFACE)
target_include_directories(GLFW_LIB SYSTEM INTERFACE "${glfw_external_BINARY_DIR}/include" "${FIXED}/include")
target_link_libraries(GLFW_LIB INTERFACE glfw ${OPENGL_glu_LIBRARY} ${GLFW_LIBRARIES})