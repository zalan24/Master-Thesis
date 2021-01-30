#include "resourcepool.h"

GenericResourcePool::ResourceId GenericResourcePool::ResourceRef::getId() const {
    return resId;
}

GenericResourcePool::ResourceRef::ResourceRef(const ResourceRef& other)
  : res(other.res), resId(other.resId), resPool(other.resPool) {
    if (resPool)
        resPool->acquire(resId);
}

GenericResourcePool::ResourceRef::ResourceRef(ResourceRef&& other)
  : res(other.res), resId(other.resId), resPool(other.resPool) {
    other.resPool = nullptr;
    other.res = nullptr;
}

GenericResourcePool::ResourceRef& GenericResourcePool::ResourceRef::operator=(
  const ResourceRef& other) {
    if (&other == this)
        return *this;
    if (resPool)
        resPool->release(resId);

    res = other.res;
    resId = other.resId;
    resPool = other.resPool;

    if (resPool)
        resPool->acquire(resId);

    return *this;
}

GenericResourcePool::ResourceRef& GenericResourcePool::ResourceRef::operator=(ResourceRef&& other) {
    if (&other == this)
        return *this;
    if (resPool)
        resPool->release(resId);

    res = other.res;
    resId = other.resId;
    resPool = other.resPool;

    other.resPool = nullptr;
    other.res = nullptr;
    return *this;
}

GenericResourcePool::ResourceRef::~ResourceRef() {
    if (resPool)
        resPool->release(resId);
}

GenericResourcePool::ResourceRef::operator bool() const {
    return res != nullptr;
}

GenericResourcePool::ResourceRef::ResourceRef() {
}

GenericResourcePool::ResourceRef::ResourceRef(const void* _res, ResourceId _resId,
                                              GenericResourcePool* _resPool)
  : res(_res), resId(_resId), resPool(_resPool) {
    resPool->acquire(resId);
}

void GenericResourcePool::registerId(const std::string& name, ResourceId id) {
    assert(resourceNames.find(name) == resourceNames.end());
    resourceNames[name] = id;
}

GenericResourcePool::ResourceId GenericResourcePool::getId(const std::string& name) {
    auto itr = resourceNames.find(name);
    if (itr == resourceNames.end())
        return INVALID_RESOURCE;
    if (!get(itr->second)) {
        resourceNames.erase(itr);
        return INVALID_RESOURCE;
    }
    return itr->second;
}

GenericResourcePool::ResourceId GenericResourcePool::getId(const std::string& name) const {
    auto itr = resourceNames.find(name);
    if (itr == resourceNames.end())
        return INVALID_RESOURCE;
    if (!has(itr->second))
        return INVALID_RESOURCE;
    return itr->second;
}
