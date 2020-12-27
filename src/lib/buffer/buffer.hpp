#pragma once

#include <vector>

#include <glad/glad.h>

template <typename T>
class Buffer
{
 public:
    Buffer(GLenum target) : bufferTarget(target) { glGenBuffers(1, &buffer); }

    Buffer(const Buffer<T>&) = delete;
    Buffer& operator=(const Buffer<T>&) = delete;

    Buffer(Buffer<T>&& other)
      : valid(other.valid), bufferTarget(other.bufferTarget), buffer(other.buffer) {
        valid = other.valid;
        buffer = other.buffer;
        bufferTarget = other.bufferTarget;
        other.valid = false;
    }
    Buffer& operator=(Buffer<T>&& other) {
        if (this == &other)
            return *this;
        valid = other.valid;
        buffer = other.buffer;
        bufferTarget = other.bufferTarget;
        other.valid = false;
        return *this;
    }

    void upload(const std::vector<T>& data, GLenum usage) const {
        glNamedBufferData(buffer, sizeof(T) * data.size(), data.data(), usage);
    }

    void uploadVertexData(const std::vector<T>& data, GLenum usage = GL_STATIC_DRAW) {
        bind();
        glBufferData(GL_ARRAY_BUFFER, sizeof(T) * data.size(), data.data(), usage);
    }

    void bind() const { glBindBuffer(bufferTarget, buffer); }

    ~Buffer() {
        if (valid)
            glDeleteBuffers(1, &buffer);
    }

 private:
    bool valid = true;
    GLenum bufferTarget;
    GLuint buffer;
};
