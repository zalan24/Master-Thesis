include(external)

download_external(
    loguru
    GIT_REPOSITORY https://github.com/emilk/loguru.git
    GIT_TAG        v2.1.0
)

file(GLOB LORUGU_FILES
  ${PROJECT_BINARY_DIR}/3rdParty/loguru/src/*.cpp
  ${PROJECT_BINARY_DIR}/3rdParty/loguru/src/*.h
  ${PROJECT_BINARY_DIR}/3rdParty/loguru/src/*.hpp
)
add_library(loguru_external STATIC ${LORUGU_FILES})
target_include_directories(loguru_external SYSTEM PUBLIC ${PROJECT_BINARY_DIR}/3rdParty/loguru/src)
