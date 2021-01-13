#pragma once

#include <vector>

#include <glad/glad.h>

template <typename T>
class Buffer
{
 public:
    Buffer() : valid(false) {}
    Buffer(GLenum target) : bufferTarget(target) { glGenBuffers(1, &buffer); }

    void reset(GLenum target) {
        close();
        valid = true;
        bufferTarget = target;
        glGenBuffers(1, &buffer);
    }

    void close() {
        if (valid)
            glDeleteBuffers(1, &buffer);
        valid = false;
    }

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
        close();
        valid = other.valid;
        buffer = other.buffer;
        bufferTarget = other.bufferTarget;
        other.valid = false;
        return *this;
    }

    void upload(const std::vector<T>& data, GLenum usage) const {
        assert(valid);
        glNamedBufferData(buffer, sizeof(T) * data.size(), data.data(), usage);
    }

    void uploadVertexData(const std::vector<T>& data, GLenum usage = GL_STATIC_DRAW) {
        assert(valid);
        bind();
        glBufferData(GL_ARRAY_BUFFER, sizeof(T) * data.size(), data.data(), usage);
        unbind();
    }

    void bind() const {
        assert(valid);
        glBindBuffer(bufferTarget, buffer);
    }
    void unbind() const {
        assert(valid);
        glBindBuffer(bufferTarget, 0);
    }
    void bind(GLuint binding) const {
        assert(valid);
        glBindBufferBase(bufferTarget, binding, buffer);
    }
    void unbind(GLuint binding) const {
        assert(valid);
        glBindBufferBase(bufferTarget, binding, 0);
    }

    ~Buffer() { close(); }

 private:
    bool valid = true;
    GLenum bufferTarget;
    GLuint buffer;
};
