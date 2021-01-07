#include "resourcemanager.h"

ResourceManager* ResourceManager::instance = nullptr;

ResourceManager::ResourceManager() {
    assert(instance == nullptr);
    instance = this;
}

ResourceManager::~ResourceManager() {
    assert(instance == this);
    instance = nullptr;
}
