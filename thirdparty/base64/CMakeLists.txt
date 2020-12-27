include(FetchContent)

FetchContent_Declare(
    base64_external
    GIT_REPOSITORY https://github.com/ReneNyffenegger/cpp-base64.git
)

FetchContent_GetProperties(base64_external)
if(NOT base64_external_POPULATED)
  FetchContent_Populate(base64_external)
endif()

file(GLOB BASE64_SOURCES ${base64_external_SOURCE_DIR}/*.cpp ${base64_external_SOURCE_DIR}/*.h ${base64_external_SOURCE_DIR}/*.hpp)
add_library(BASE64_LIB ${BASE64_SOURCES})
target_include_directories(BASE64_LIB SYSTEM PUBLIC "${base64_external_BINARY_DIR}" "${base64_external_SOURCE_DIR}")
