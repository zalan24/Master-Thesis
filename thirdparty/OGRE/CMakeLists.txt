include(FetchContent)

FetchContent_Declare(
    ogre_external
    GIT_REPOSITORY https://github.com/OGRECave/ogre.git
)

FetchContent_GetProperties(ogre_external)
if(NOT ogre_external_POPULATED)
  FetchContent_Populate(ogre_external)
endif()

set(OGRE_BUILD_RENDERSYSTEM_D3D9 OFF)
add_subdirectory(${ogre_external_SOURCE_DIR} ${ogre_external_BINARY_DIR})

add_library(OGRE_LIB INTERFACE)
add_dependencies(OGRE_LIB ogre_external)
# target_include_directories(OGRE_LIB SYSTEM INTERFACE ${ogre_external_SOURCE_DIR})
