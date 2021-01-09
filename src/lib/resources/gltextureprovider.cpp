#include "gltextureprovider.h"

#include <memory>

#include <loadimage.h>

#include "resourcemanager.h"

GlTextureProvider::GlTextureProvider(
  ResourcePool<GlTexture, TextureProvider::ResourceDescriptor>* _texPool)
  : texPool(_texPool) {
}

GenericResourcePool::ResourceRef GlTextureProvider::getResource(
  const ResourceDescriptor& desc) const {
    GenericResourcePool::ResourceId resId = texPool->getDescId(desc);
    if (resId != GenericResourcePool::INVALID_RESOURCE)
        return texPool->get(resId);
    GenericResourcePool::ResourceRef res;
    if (desc.isColor()) {
        RGBA color;
        glm::vec4 value = desc.getColor();
        color.set(value.r, value.g, value.b, value.a);
        res = static_cast<const TextureProvider*>(this)->createResource(color);
    }
    else if (desc.isFile())
        res = createResource(desc.getFilename());
    else
        assert(false);
    texPool->registerDesc(desc, res.getId());
    return res;
}

GenericResourcePool::ResourceRef GlTextureProvider::createResource(const Texture<RGBA>& tex) const {
    std::unique_ptr<GlTexture> glTex = std::make_unique<GlTexture>(GL_TEXTURE_2D, GL_RGBA);
    glTex->upload(&tex);
    return texPool->add(std::move(glTex));
}

GenericResourcePool::ResourceRef GlTextureProvider::createResource(
  const std::string& filename) const {
    Texture<RGBA> tex = load_image<RGBA>(filename);
    return createResource(tex);
}
