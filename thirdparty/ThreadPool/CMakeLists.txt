include(external)

download_external(
    thread_pool
    GIT_REPOSITORY https://github.com/progschj/ThreadPool.git
)

add_library(thread_pool_external INTERFACE)
target_include_directories(thread_pool_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/thread_pool/src)
