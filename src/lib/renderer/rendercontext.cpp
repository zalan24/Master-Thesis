#include "rendercontext.h"

#include <stdexcept>

#include <glad/glad.h>

void checkError() {
    GLenum error = glGetError();
    if (error == GL_NO_ERROR)
        return;
    throw std::runtime_error("GL error");
}
