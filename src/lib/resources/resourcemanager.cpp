#include "resourcemanager.h"

#include "glmeshprovider.h"
#include "gltextureprovider.h"

ResourceManager* ResourceManager::instance = nullptr;

ResourceManager::ResourceManager(ResourceInfos resource_infos)
  : resourceInfos(std::move(resource_infos)) {
    assert(instance == nullptr);
    instance = this;
    textureProvider =
      std::make_unique<GlTextureProvider>(resourceInfos.resourceFolder, &glTexturePool);
    meshProvider = std::make_unique<GlMeshProvider>(resourceInfos.resourceFolder,
                                                    resourceInfos.modelResourcesJson,
                                                    textureProvider.get(), &glMeshPool);
}

ResourceManager::~ResourceManager() {
    assert(instance == this);
    instance = nullptr;
}
