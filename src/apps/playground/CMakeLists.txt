
add_executable(Playground playground.cpp)
target_link_libraries(Playground PRIVATE
    AnimationsEngine
    AnimationsController
    AnimationsRenderer
    CLI11_LIB
)
