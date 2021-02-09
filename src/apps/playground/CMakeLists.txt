
add_executable(Playground playground.cpp)
target_link_libraries(Playground PRIVATE
    AnimationsEngine
    AnimationsController
    AnimationsRenderer
    cli11_external
)
