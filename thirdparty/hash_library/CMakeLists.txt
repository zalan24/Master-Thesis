# include(FetchContent)

# FetchContent_Declare(
#     hash_library_external
#     GIT_REPOSITORY https://github.com/stbrumme/hash-library.git
# )

# FetchContent_GetProperties(hash_library_external)
# if(NOT hash_library_externalPOPULATED)
#   FetchContent_Populate(hash_library_external)
#   prep_thirdparty(${hash_library_external_SOURCE_DIR} "${hash_library_external_SOURCE_DIR}_fixed")
# endif()
# set(FIXED "${hash_library_external_SOURCE_DIR}_fixed")

include(external)

download_external(
    hash_library
    GIT_REPOSITORY https://github.com/stbrumme/hash-library.git
)

add_library(hash_library_external INTERFACE)
target_include_directories(hash_library_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/hash_library/src)

file(GLOB FILEINFO_FILES
  ${PROJECT_BINARY_DIR}/3rdParty/hash_library/src/*.cpp
  ${PROJECT_BINARY_DIR}/3rdParty/hash_library/src/*.h
  ${PROJECT_BINARY_DIR}/3rdParty/hash_library/src/*.hpp
)
add_library(HASH_LIBRARY_LIB STATIC ${FILEINFO_FILES})
target_link_libraries(HASH_LIBRARY_LIB PUBLIC
  hash_library_external
)
