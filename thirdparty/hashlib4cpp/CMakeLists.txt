include(external)

download_external(
    hashlib4cpp
    GIT_REPOSITORY https://github.com/ron4fun/HashLib4CPP.git
)

add_library(hashlib4cpp_external INTERFACE)
target_include_directories(hashlib4cpp_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/hashlib4cpp/src)
