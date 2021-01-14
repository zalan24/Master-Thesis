#include "framebuffer.h"

Framebuffer::Framebuffer() {
    glGenFramebuffers(1, &fbo);
}

void Framebuffer::close() {
    if (fbo == 0)
        return;
    glDeleteFramebuffers(1, &fbo);
    color.close();
    depthStencil.close();
    fbo = 0;
}

Framebuffer::~Framebuffer() {
    close();
}

Framebuffer::Framebuffer(Framebuffer&& other)
  : fbo(other.fbo), color(std::move(other.color)), depthStencil(std::move(other.depthStencil)) {
    other.fbo = 0;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) {
    if (this == &other)
        return *this;
    close();
    fbo = other.fbo;
    color = std::move(other.color);
    depthStencil = std::move(other.depthStencil);

    other.fbo = 0;
    return *this;
}

void Framebuffer::bind(GLenum bindMode) const {
    glBindFramebuffer(bindMode, fbo);
}

void Framebuffer::unbind(GLenum bindMode) const {
    glBindFramebuffer(bindMode, 0);
}

bool Framebuffer::isBound(GLenum bindMode) const {
    GLint id;
    glGetIntegerv(bindMode, &id);
    return id == fbo;
}

bool Framebuffer::setResolution(unsigned int _width, unsigned int _height) {
    if (width == _width && height == _height)
        return false;
    width = _width;
    height = _height;
    return true;
}

void Framebuffer::createColor() {
    assert(isBound());
    GlTexture::SamplingState sampling;
    sampling.wrapX = GL_CLAMP_TO_EDGE;
    sampling.wrapY = GL_CLAMP_TO_EDGE;
    sampling.magFilter = GL_LINEAR;
    sampling.minFilter = GL_LINEAR;
    color = GlTexture(GL_TEXTURE_2D, GL_RGB, sampling);
    color.create(width, height, 1, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    color.bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color.getId(), 0);
    color.unbind();
}

void Framebuffer::createDepth() {
    assert(isBound());
    GlTexture::SamplingState sampling;
    sampling.wrapX = GL_CLAMP_TO_EDGE;
    sampling.wrapY = GL_CLAMP_TO_EDGE;
    sampling.magFilter = GL_NEAREST;
    sampling.minFilter = GL_NEAREST;
    depthStencil = GlTexture(GL_TEXTURE_2D, GL_DEPTH_COMPONENT, sampling);
    depthStencil.create(width, height, 1, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

    depthStencil.bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthStencil.getId(),
                           0);
    depthStencil.unbind();
}

void Framebuffer::createDepthStencil() {
    assert(isBound());
    GlTexture::SamplingState sampling;
    sampling.wrapX = GL_CLAMP_TO_EDGE;
    sampling.wrapY = GL_CLAMP_TO_EDGE;
    sampling.magFilter = GL_NEAREST;
    sampling.minFilter = GL_NEAREST;
    depthStencil = GlTexture(GL_TEXTURE_2D, GL_DEPTH24_STENCIL8, sampling);
    depthStencil.create(width, height, 1, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, nullptr);

    depthStencil.bind();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                           depthStencil.getId(), 0);
    depthStencil.unbind();
}

bool Framebuffer::isComplete() const {
    assert(isBound());
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    return status == GL_FRAMEBUFFER_COMPLETE;
}
