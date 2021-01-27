include(FetchContent)

FetchContent_Declare(
    concurrent_queue_external
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        1.0.2
)

FetchContent_GetProperties(concurrent_queue_external)
if(NOT concurrent_queue_externalPOPULATED)
  FetchContent_Populate(concurrent_queue_external)
  prep_thirdparty(${concurrent_queue_external_SOURCE_DIR} "${concurrent_queue_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${concurrent_queue_external_SOURCE_DIR}_fixed")

add_library(CONCURRENT_QUEUE_LIB INTERFACE)
add_dependencies(CONCURRENT_QUEUE_LIB concurrent_queue_external)
target_include_directories(CONCURRENT_QUEUE_LIB SYSTEM INTERFACE ${FIXED})
