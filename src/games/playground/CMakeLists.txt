add_subdirectory(shaders)
add_subdirectory(game)


add_executable(Playground playground.cpp)
target_link_libraries(Playground PRIVATE
    AnimationsEngine
    AnimationsController
    PlaygroundLib
    cli11_external
)
add_dependencies(Playground PlaygroundShaderBinary)
