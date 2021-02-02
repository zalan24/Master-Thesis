include(FetchContent)

FetchContent_Declare(
    simplecpp_external
    GIT_REPOSITORY https://github.com/danmar/simplecpp.git
)

FetchContent_GetProperties(simplecpp_external)
if(NOT simplecpp_external_POPULATED)
  FetchContent_Populate(simplecpp_external)
  prep_thirdparty(${simplecpp_external_SOURCE_DIR} "${simplecpp_external_SOURCE_DIR}_fixed")
  add_subdirectory("${simplecpp_external_SOURCE_DIR}_fixed" ${simplecpp_external_BINARY_DIR})
endif()
set(FIXED "${simplecpp_external_SOURCE_DIR}_fixed")

add_library(SimpleCpp_LIB STATIC ${FIXED}/simplecpp.cpp)
set_target_properties(  SimpleCpp_LIB
                        PROPERTIES
                        COMPILE_FLAGS ${TARGETCOMPILEFLAGS}
                        LINK_OPTIONS "${TARGET_LINK_FLAGS}")
target_include_directories(SimpleCpp_LIB SYSTEM PUBLIC "${FIXED}")
