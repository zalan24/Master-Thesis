include(external)

download_external(
    concurrent_queue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG        1.0.2
)

add_library(concurrent_queue_external INTERFACE)
target_include_directories(concurrent_queue_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/concurrent_queue/src)

