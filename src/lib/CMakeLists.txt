add_subdirectory(data)
add_subdirectory(io)
add_subdirectory(tools)
add_subdirectory(window)
add_subdirectory(renderer)
add_subdirectory(shaders)
add_subdirectory(input)
add_subdirectory(buffer)
add_subdirectory(entities)
add_subdirectory(game)
add_subdirectory(engine)
add_subdirectory(serializable)

add_library(AnimationsLibs INTERFACE)
target_link_libraries(AnimationsLibs INTERFACE
                    AnimationsData
                    AnimationsIO
                    AnimationsTools
                    AnimationsWindow
                    AnimationsRenderer
                    AnimationsShaderManager
                    AnimationsEntities
                    AnimationsGame
                    AnimationsEngine
                    AnimationsSerializable
)
