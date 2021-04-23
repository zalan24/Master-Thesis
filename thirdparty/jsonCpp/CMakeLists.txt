include(external)

# Beacuse VulkanTool needs it and it scks too much to download it itself...

download_external(
    json_cpp
    GIT_REPOSITORY https://github.com/Kurento/jsoncpp.git
    GIT_TAG        1.6.4
)


external(
    json_cpp
    INSTALL ON
    TARGETS
    PROJECT_OPTIONS
        -DJSONCPP_WITH_TESTS=OFF
        -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF
)
