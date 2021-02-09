include(external)

download_external(
    mapbox
    GIT_REPOSITORY https://github.com/mapbox/eternal.git
)

add_library(mapbox_external INTERFACE)
target_include_directories(mapbox_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/mapbox/src/include)
