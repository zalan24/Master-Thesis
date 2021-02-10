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
