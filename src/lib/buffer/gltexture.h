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

    GlTexture(GLenum target, GLint internalformat, const SamplingState& sampling = SamplingState());
    ~GlTexture();

    void setSampling(const SamplingState& sampling);
    const SamplingState& getSampling() const;

    template <typename P>
    void upload(const Texture<P>* texture, bool genMips = true) {
        bind();
        static constexpr GLenum format = PixelType<P>::format;
        static constexpr GLenum dataType = PixelType<P>::dataType;
        if (is1D()) {
            assert(texture->getHeight() == 1 && texture->getDepth() == 1);
            glTexImage1D(target, 0, internalFormat, texture->getWidth(), 0, format, dataType,
                         texture->getData());
        }
        else if (is3D()) {
            // 3D
            glTexImage3D(target, 0, internalFormat, texture->getWidth(), texture->getHeight(),
                         texture->getDepth(), 0, format, dataType, texture->getData());
        }
        else {
            // 2D
            assert(is2D());
            assert(texture->getDepth() == 1);
            glTexImage2D(target, 0, internalFormat, texture->getWidth(), texture->getHeight(), 0,
                         format, dataType, texture->getData());
        }
        uploaded = true;
        setParams();
        if (genMips)
            glGenerateMipmap(target);
        unbind();
    }

    void bind() const;
    void unbind() const;

    bool is1D() const;
    bool is2D() const;
    bool is3D() const;

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
