include(external)

download_external(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.5
)

add_library(glm_external INTERFACE)
target_include_directories(glm_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/glm/src)
