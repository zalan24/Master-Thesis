include(FetchContent)

FetchContent_Declare(
    base64_external
    GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
)

FetchContent_GetProperties(base64_external)
if(NOT base64_external_POPULATED)
  FetchContent_Populate(base64_external)
  prep_thirdparty(${base64_external_SOURCE_DIR} "${base64_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${base64_external_SOURCE_DIR}_fixed")

set(BASE64_SOURCES ${FIXED}/base64.cpp ${FIXED}/base64.h)
add_library(BASE64_LIB STATIC ${BASE64_SOURCES})
target_include_directories(BASE64_LIB SYSTEM PUBLIC "${base64_external_BINARY_DIR}" "${FIXED}")
