include(FetchContent)

FetchContent_Declare(
    thread_pool_external
    GIT_REPOSITORY https://github.com/progschj/ThreadPool.git
)

FetchContent_GetProperties(thread_pool_external)
if(NOT thread_pool_external_POPULATED)
  FetchContent_Populate(thread_pool_external)
endif()
add_library(ThreadPool_LIB INTERFACE)
add_dependencies(ThreadPool_LIB thread_pool_external)
target_include_directories(ThreadPool_LIB SYSTEM INTERFACE ${thread_pool_external_SOURCE_DIR})
