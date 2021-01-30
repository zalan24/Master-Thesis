include(FetchContent)

FetchContent_Declare(
    hash_library_external
    GIT_REPOSITORY https://github.com/stbrumme/hash-library.git
)

FetchContent_GetProperties(hash_library_external)
if(NOT hash_library_externalPOPULATED)
  FetchContent_Populate(hash_library_external)
  prep_thirdparty(${hash_library_external_SOURCE_DIR} "${hash_library_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${hash_library_external_SOURCE_DIR}_fixed")

add_library(HASH_LIBRARY_INTERFACE INTERFACE)
add_dependencies(HASH_LIBRARY_INTERFACE hash_library_external)
file(GLOB FILEINFO_FILES
  ${FIXED}/*.cpp
  ${FIXED}/*.h
  ${FIXED}/*.hpp
)
add_library(HASH_LIBRARY_LIB STATIC ${FILEINFO_FILES})
target_link_libraries(HASH_LIBRARY_LIB PRIVATE
  HASH_LIBRARY_INTERFACE
)
target_include_directories(HASH_LIBRARY_LIB SYSTEM INTERFACE ${FIXED})
