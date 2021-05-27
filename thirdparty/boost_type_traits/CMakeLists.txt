include(external)

download_external(
    boost_type_traits
    GIT_REPOSITORY https://github.com/boostorg/type_traits.git
    GIT_TAG        boost-1.75.0
)

add_library(boost_type_traits_external INTERFACE)
target_include_directories(boost_type_traits_external SYSTEM INTERFACE ${PROJECT_BINARY_DIR}/3rdParty/boost_type_traits/src/include)
target_link_libraries(boost_type_traits_external INTERFACE boost_config_external boost_assert_external boost_static_assert_external)
