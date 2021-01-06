#include "gltexture.h"

GlTexture::GlTexture(GLenum _target, GLint _internalformat, const SamplingState& _sampling)
  : target(_target), internalFormat(_internalformat), sampling(_sampling) {
    glGenTextures(1, &textureID);
}

GlTexture::~GlTexture() {
    glDeleteTextures(1, &textureID);
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
