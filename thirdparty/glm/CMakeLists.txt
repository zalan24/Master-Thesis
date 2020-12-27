include(FetchContent)

FetchContent_Declare(
    glm_external
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.5
)

FetchContent_GetProperties(glm_external)
if(NOT glm_external_POPULATED)
  FetchContent_Populate(glm_external)
endif()
add_library(GLM_LIB INTERFACE)
add_dependencies(GLM_LIB glm_external)
target_include_directories(GLM_LIB SYSTEM INTERFACE ${glm_external_SOURCE_DIR})
