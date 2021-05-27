include(external)

download_external(
    flexible_reflection
    GIT_REPOSITORY https://github.com/preshing/FlexibleReflection.git
    GIT_TAG        part1
)

add_library(flexible_reflection_external INTERFACE)
target_include_directories(flexible_reflection_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/flexible_reflection/src)
