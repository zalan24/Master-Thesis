#pragma once

#include <hash_map>
#include <limits>
#include <memory>
#include <string>

template <typename R>
class ResourcePool
{
 public:
    using ResourceId = size_t;
    static constexpr ResourceId INVALID_RESOURCE = std::numeric_limits<ResourceId>::max();

    class ResourceRef
    {
     public:
        const R* getRes() const { return res; }
        ResourceId getId() const { return resId; }

        ResourceRef(const ResourceRef& other)
          : res(other.res), resId(other.resId), resPool(other.resPool) {
            if (resPool)
                resPool->acquire(resId);
        }

        ResourceRef(ResourceRef&& other)
          : res(other.res), resId(other.resId), resPool(other.resPool) {
            other.resPool = nullptr;
        }

        ~ResourceRef() {
            if (resPool)
                resPool->release(resId);
        }

     private:
        ResourceRef(const R* _res, ResourceId _resId, ResourcePool<R>* _resPool)
          : res(_res), resId(_resId), resPool(_resPool) {
            resPool->acquire(resId);
        }

        const R* res;
        ResourceId resId;
        ResourcePool<R>* resPool;
    };

    ResourceRef get(ResourceId id) {
        auto itr = resources.find(id);
        assert(itr != resources.end());
        return ResourceRef(itr->second.res.get(), id, this);
    }

    ResourceId getId(const std::string& name) {
        auto itr = resourceNames.find(name);
        if (itr == resourceNames.end())
            return INVALID_RESOURCE;
        auto resItr = resources.find(itr->second);
        if (resItr == resources.end()) {
            resourceNames.erase(itr);
            return INVALID_RESOURCE;
        }
        return itr->second;
    }

    ResourceRef add(std::unique_ptr<R>&& res) {
        ResourceId id = currentId++;
        resources[id] = {std::move(res), 0};
        return get(id);
    }

    ResourceRef add(const std::string& name, std::unique_ptr<R>&& res) {
        assert(resourceNames.find(name) == resourceNames.end());
        ResourceRef ret = add(std::move(res));
        resourceNames[name] = ret.getId();
        return ret;
    }

 private:
    struct ResourceData
    {
        std::unique_ptr<R> res;
        size_t refCount;
    };
    std::hash_map<std::string, ResourceId> resourceNames;
    std::hash_map<ResourceId, ResourceData> resources;
    ResourceId currentId = 0;

    void acquire(ResourceId id) { resources[id].refCount++; }

    void release(ResourceId id) {
        auto itr = resources.find(id);
        assert(itr != resources.end());
        if (--itr->second.refCount == 0)
            resources.erase(itr);
    }
};
