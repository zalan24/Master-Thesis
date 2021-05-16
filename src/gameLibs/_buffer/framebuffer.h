#pragma once

#include <glad/glad.h>

#include "gltexture.h"

class Framebuffer
{
 public:
    Framebuffer();
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& other);
    Framebuffer& operator=(Framebuffer&& other);

    void bind(GLenum bindMode = GL_FRAMEBUFFER) const;
    void unbind(GLenum bindMode = GL_FRAMEBUFFER) const;
    bool isBound(GLenum bindMode = GL_FRAMEBUFFER_BINDING) const;

    void close();

    void createColor();
    void createDepth();
    void createDepthStencil();

    // Returns true if need to recreate attachments
    bool setResolution(unsigned int width, unsigned int height);

    unsigned int getCurrentWidth() const { return width; }
    unsigned int getCurrentHeight() const { return height; }

    GLuint getFramebufferObject() const { return fbo; }

    bool isComplete() const;

 private:
    GLuint fbo = 0;
    GlTexture color;
    GlTexture depthStencil;
    unsigned int width = 0;
    unsigned int height = 0;
};
