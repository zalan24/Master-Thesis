include(FetchContent)

FetchContent_Declare(
    mapbox_external
    GIT_REPOSITORY https://github.com/mapbox/eternal.git
)

FetchContent_GetProperties(mapbox_external)
if(NOT mapbox_externalPOPULATED)
  FetchContent_Populate(mapbox_external)
endif()
add_library(MAPBOX_LIB INTERFACE)
add_dependencies(MAPBOX_LIB mapbox_external)
target_include_directories(MAPBOX_LIB SYSTEM INTERFACE ${mapbox_external_SOURCE_DIR}/include)
