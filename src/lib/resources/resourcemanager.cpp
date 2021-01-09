#include "resourcemanager.h"

#include "gltextureprovider.h"

ResourceManager* ResourceManager::instance = nullptr;

ResourceManager::ResourceManager() {
    assert(instance == nullptr);
    instance = this;
    textureProvider = std::make_unique<GlTextureProvider>(&glTexturePool);
}

ResourceManager::~ResourceManager() {
    assert(instance == this);
    instance = nullptr;
}
