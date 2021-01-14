#include "gltexture.h"

GlTexture::GlTexture(GLenum _target, GLint _internalformat, const SamplingState& _sampling)
  : target(_target), internalFormat(_internalformat), sampling(_sampling) {
    glGenTextures(1, &textureID);
}

GlTexture::GlTexture() {
}

void GlTexture::close() {
    if (textureID == 0)
        return;
    glDeleteTextures(1, &textureID);
    textureID = 0;
    uploaded = false;
}

GlTexture::operator bool() const {
    return textureID > 0;
}

GlTexture::GlTexture(GlTexture&& other)
  : target(other.target),
    internalFormat(other.internalFormat),
    sampling(std::move(other.sampling)),
    textureID(other.textureID),
    uploaded(other.uploaded) {
    other.textureID = 0;
}

GlTexture& GlTexture::operator=(GlTexture&& other) {
    if (this == &other)
        return *this;
    close();
    target = other.target;
    internalFormat = other.internalFormat;
    sampling = std::move(other.sampling);
    textureID = other.textureID;
    uploaded = other.uploaded;

    other.textureID = 0;
    return *this;
}

GlTexture::~GlTexture() {
    close();
}

void GlTexture::bind() const {
    glBindTexture(target, textureID);
}

void GlTexture::unbind() const {
    glBindTexture(target, 0);
}

void GlTexture::setParams() const {
    glTexParameteri(target, GL_TEXTURE_WRAP_S, sampling.wrapX);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, sampling.wrapY);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, sampling.magFilter);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, sampling.minFilter);
}

bool GlTexture::is1D() const {
    return target == GL_TEXTURE_1D || target == GL_PROXY_TEXTURE_1D;
}

bool GlTexture::is2D() const {
    return !is1D() && !is3D();
}

bool GlTexture::is3D() const {
    return target == GL_TEXTURE_3D || target == GL_PROXY_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY
           || target == GL_PROXY_TEXTURE_2D_ARRAY;
}

void GlTexture::setSampling(const SamplingState& _sampling) {
    sampling = _sampling;
    if (uploaded) {
        bind();
        setParams();
        unbind();
    }
}

const GlTexture::SamplingState& GlTexture::getSampling() const {
    return sampling;
}

void GlTexture::create(unsigned int width, unsigned int height, unsigned int depth, GLenum format,
                       GLenum dataType, const void* pixels) {
    bind();
    if (is1D()) {
        assert(height == 1 && depth == 1);
        glTexImage1D(target, 0, internalFormat, width, 0, format, dataType, pixels);
    }
    else if (is3D()) {
        // 3D
        glTexImage3D(target, 0, internalFormat, width, height, depth, 0, format, dataType, pixels);
    }
    else {
        // 2D
        assert(is2D());
        assert(depth == 1);
        glTexImage2D(target, 0, internalFormat, width, height, 0, format, dataType, pixels);
    }
    uploaded = true;
    setParams();
    unbind();
}
