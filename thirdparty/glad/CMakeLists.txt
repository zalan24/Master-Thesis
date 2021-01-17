add_library(GLAD_LIB STATIC src/glad.c)
target_include_directories(GLAD_LIB SYSTEM PUBLIC include)