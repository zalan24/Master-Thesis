include(external)

download_external(
    robin_hood_hash
    GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
    GIT_TAG        3.11.1
)

external(
    robin_hood_hash
    INSTALL ON
    PROJECT_OPTIONS
        -DRH_STANDALONE_PROJECT=OFF
)
