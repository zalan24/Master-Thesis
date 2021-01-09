#pragma once

#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

class GenericResourcePool
{
 public:
    using ResourceId = size_t;
    static constexpr ResourceId INVALID_RESOURCE = std::numeric_limits<ResourceId>::max();

    class ResourceRef
    {
     public:
        ResourceRef();
        ResourceRef(const ResourceRef& other);
        ResourceRef(ResourceRef&& other);
        ResourceRef& operator=(const ResourceRef& other);
        ResourceRef& operator=(ResourceRef&& other);
        ~ResourceRef();

        GenericResourcePool::ResourceId getId() const;
        operator bool() const;

        template <typename R>
        const R* getRes() const {
            assert(resPool && res);
            assert(resPool->resTypeInfo == typeid(R));
            return static_cast<const R*>(res);
        }

     private:
        ResourceRef(const void* res, GenericResourcePool::ResourceId resId,
                    GenericResourcePool* resPool);

        const void* res = nullptr;
        GenericResourcePool::ResourceId resId = INVALID_RESOURCE;
        GenericResourcePool* resPool = nullptr;

        template <typename R, typename D>
        friend class ResourcePool;
    };

    GenericResourcePool(const GenericResourcePool&) = delete;
    GenericResourcePool& operator=(const GenericResourcePool&) = delete;

    void registerId(const std::string& name, ResourceId id);
    ResourceId getId(const std::string& name);
    ResourceId getId(const std::string& name) const;

    virtual ResourceRef get(ResourceId id) = 0;
    virtual bool has(ResourceId id) const = 0;

 protected:
    template <typename R>
    GenericResourcePool() : resTypeInfo(typeid(R)) {}

    ~GenericResourcePool() = default;

    virtual void acquire(ResourceId id) = 0;
    virtual void release(ResourceId id) = 0;

 private:
    std::type_info resTypeInfo;
    std::unordered_map<std::string, ResourceId> resourceNames;
};

template <typename R, typename D>
class ResourcePool final : public GenericResourcePool
{
 public:
    ResourcePool() : GenericResourcePool<R>() {}
    ~ResourcePool() { assert(resources.empty()); }

    ResourceRef get(ResourceId id) override {
        auto itr = resources.find(id);
        assert(itr != resources.end());
        return ResourceRef(static_cast<const void*>(itr->second.res.get()), id, this);
    }

    bool has(ResourceId id) const override {
        auto itr = resources.find(id);
        return itr != resources.end();
    }

    ResourceRef add(std::unique_ptr<R>&& res) {
        ResourceId id = currentId++;
        resources[id] = {std::move(res), 0};
        return get(id);
    }

    void registerDesc(const D& desc, ResourceId id) {
        assert(resourceDesc.find(decs) == resourceDesc.end());
        resourceDesc[decs] = id;
    }

    ResourceId getDescId(const D& desc) {
        auto itr = resourceDesc.find(desc);
        if (itr == resourceDesc.end())
            return INVALID_RESOURCE;
        if (!get(itr->second)) {
            resourceDesc.erase(itr);
            return INVALID_RESOURCE;
        }
        return itr->second;
    }

    ResourceId getDescId(const D& desc) const {
        auto itr = resourceDesc.find(desc);
        if (itr == resourceDesc.end())
            return INVALID_RESOURCE;
        if (!get(itr->second))
            return INVALID_RESOURCE;
        return itr->second;
    }

 protected:
    void acquire(ResourceId id) override { resources[id].refCount++; }

    void release(ResourceId id) override {
        auto itr = resources.find(id);
        assert(itr != resources.end());
        if (--itr->second.refCount == 0)
            resources.erase(itr);
    }

 private:
    struct ResourceData
    {
        std::unique_ptr<R> res;
        size_t refCount;
    };
    std::unordered_map<ResourceId, ResourceData> resources;
    ResourceId currentId = 0;
    std::unordered_map<D, ResourceId> resourceDesc;
};
