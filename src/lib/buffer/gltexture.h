#pragma once

#include <cassert>

#include <glad/glad.h>

#include <pixel.h>
#include <texture.hpp>

class GlTexture
{
 public:
    struct SamplingState
    {
        GLint wrapX = GL_REPEAT;
        GLint wrapY = GL_REPEAT;
        GLint magFilter = GL_LINEAR;
        GLint minFilter = GL_LINEAR_MIPMAP_LINEAR;
        SamplingState() noexcept {}
    };

    GlTexture();
    GlTexture(GLenum target, GLint internalformat, const SamplingState& sampling = SamplingState());
    ~GlTexture();

    GlTexture(const GlTexture&) = delete;
    GlTexture& operator=(const GlTexture&) = delete;
    GlTexture(GlTexture&& other);
    GlTexture& operator=(GlTexture&& other);

    operator bool() const;

    void close();

    void setSampling(const SamplingState& sampling);
    const SamplingState& getSampling() const;

    void create(unsigned int width, unsigned int height, unsigned int depth, GLenum format,
                GLenum dataType, const void* pixels = nullptr);

    template <typename P>
    void upload(const Texture<P>* texture, bool genMips = true) {
        static constexpr GLenum format = PixelType<P>::format;
        static constexpr GLenum dataType = PixelType<P>::dataType;
        create(static_cast<unsigned int>(texture->getWidth()),
               static_cast<unsigned int>(texture->getHeight()),
               static_cast<unsigned int>(texture->getDepth()), format, dataType,
               texture->getData());
        if (genMips) {
            bind();
            glGenerateMipmap(target);
            unbind();
        }
    }

    void bind() const;
    void unbind() const;

    bool is1D() const;
    bool is2D() const;
    bool is3D() const;

    GLuint getId() const { return textureID; }

 private:
    GLenum target;
    GLint internalFormat;
    SamplingState sampling;
    GLuint textureID;
    bool uploaded = false;

    void setParams() const;

    template <typename P>
    struct PixelType;

    template <>
    struct PixelType<RGBA>
    {
        static constexpr GLenum format = GL_RGBA;
        static constexpr GLenum dataType = GL_UNSIGNED_INT_8_8_8_8;
    };
};
