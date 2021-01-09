#include "resourcemanager.h"

#include "glmeshprovider.h"
#include "gltextureprovider.h"

ResourceManager* ResourceManager::instance = nullptr;

ResourceManager::ResourceManager() {
    assert(instance == nullptr);
    instance = this;
    textureProvider = std::make_unique<GlTextureProvider>(&glTexturePool);
    meshProvider = std::make_unique<GlMeshProvider>(textureProvider.get(), &glMeshPool);
}

ResourceManager::~ResourceManager() {
    assert(instance == this);
    instance = nullptr;
}
