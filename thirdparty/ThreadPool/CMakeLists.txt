include(FetchContent)

FetchContent_Declare(
    thread_pool_external
    GIT_REPOSITORY https://github.com/progschj/ThreadPool.git
)

FetchContent_GetProperties(thread_pool_external)
if(NOT thread_pool_external_POPULATED)
  FetchContent_Populate(thread_pool_external)
  prep_thirdparty(${thread_pool_external_SOURCE_DIR} "${thread_pool_external_SOURCE_DIR}_fixed")
endif()
set(FIXED "${thread_pool_external_SOURCE_DIR}_fixed")

add_library(ThreadPool_LIB INTERFACE)
add_dependencies(ThreadPool_LIB thread_pool_external)
target_include_directories(ThreadPool_LIB SYSTEM INTERFACE ${FIXED})
