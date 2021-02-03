include(FetchContent)

FetchContent_Declare(
    hashlib4cpp_external
    GIT_REPOSITORY https://github.com/ron4fun/HashLib4CPP.git
)

FetchContent_GetProperties(hashlib4cpp_external)
if(NOT hashlib4cpp_externalPOPULATED)
  FetchContent_Populate(hashlib4cpp_external)
  prep_thirdparty(${hashlib4cpp_external_SOURCE_DIR} "${hashlib4cpp_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${hashlib4cpp_external_SOURCE_DIR}_fixed")

add_library(HashLib4CPP_LIB INTERFACE)
add_dependencies(HashLib4CPP_LIB hashlib4cpp_external)
target_include_directories(HashLib4CPP_LIB SYSTEM INTERFACE ${FIXED})
