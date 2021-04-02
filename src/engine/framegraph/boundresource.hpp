#pragma once

// This class is meant to be used for resources like image views,
// where an object needs to be created for a parent resource and it's bound to it
// The goal is to handle the case, where the parent object gets recreated

#include <mutex>
#include <type_traits>
#include <vector>

#include <drverror.h>

class BoundResourceParent;

class BoundResourceBase
{
 public:
    // parent: the parent resource that triggered the destruction
    virtual void destroy(BoundResourceParent* parent) = 0;

 protected:
    ~BoundResourceBase() {}
};

struct BindingData
{
    BoundResourceBase* resource = nullptr;
    std::mutex m;
};

struct BindingParentData
{
    BoundResourceParent* parent = nullptr;
    std::mutex m;
};

class BoundResourceParent
{
 public:
    BoundResourceParent() {
        binding = new BindingParentData;
        binding->parent = this;
    }
    BoundResourceParent(const BoundResourceParent&) = delete;
    BoundResourceParent& operator=(const BoundResourceParent&) = delete;
    BoundResourceParent(BoundResourceParent&& other) : binding(other.binding) {
        if (binding) {
            std::unique_lock<std::mutex> lock(binding->m);
            bindings = std::move(other.bindings);
            binding->parent = this;
        }
        else
            bindings = std::move(other.bindings);
        other.binding = nullptr;
    }
    BoundResourceParent& operator=(BoundResourceParent&& other) {
        if (&other == this)
            return *this;
        close();
        binding = other.binding;
        if (binding) {
            std::unique_lock<std::mutex> lock(binding->m);
            bindings = std::move(other.bindings);
            binding->parent = this;
        }
        else
            bindings = std::move(other.bindings);
        other.binding = nullptr;
        return *this;
    }
    ~BoundResourceParent() { close(); }

    void destroyBindings() {
        std::unique_lock<std::mutex> lock(m);
        for (BindingData* b : bindings) {
            std::unique_lock<std::mutex> lock(b->m);
            b->resource->destroy(this);
        }
        bindings.clear();
    }

    BindingParentData* bind(BindingData* b) {
        std::unique_lock<std::mutex> lock(m);
        bindings.push_back(b);
        return binding;
    }

    void unbind(BindingData* b) {
        std::unique_lock<std::mutex> lock(m);
        bindings.erase(std::find(bindings.begin(), bindings.end(), b));
    }

 private:
    BindingParentData* binding = nullptr;
    std::mutex m;
    std::vector<BindingData*> bindings;

    void close() {
        if (binding) {
            std::unique_lock<std::mutex> lock(binding->m);
            destroyBindings();
            delete binding;
            binding = nullptr;
        }
        else
            drv::drv_assert(bindings.size() == 0);
    }
};

template <typename T, typename D>
class BoundResource : public BoundResourceBase
{
 public:
    using Gen = T(const D&, uint32_t,
                  const BoundResourceParent**);  // generate new value from parents
    using Del = void(const D&, T&&);             // delete current value

    BoundResource(D&& _data, Gen&& _gen, Del&& _del)
      : data(std::move(_data)), gen(std::move(_gen)), del(std::move(_del)) {
        binding = new BindingData;
        binding->resource = this;
    }

    BoundResource(const BoundResource&) = delete;
    BoundResource& operator=(const BoundResource&) = delete;

    BoundResource(BoundResource&& other) : binding(other.binding) {
        if (binding) {
            std::unique_lock<std::mutex> lock(binding->m);
            parents = std::move(other.parents);
            value = std::move(other.value);
            data = std::move(other.data);
            gen = std::move(other.gen);
            del = std::move(other.del);
            hasResource = other.hasResource;
            binding->resource = this;
        }
        else {
            parents = std::move(other.parents);
            value = std::move(other.value);
            data = std::move(other.data);
            gen = std::move(other.gen);
            del = std::move(other.del);
            hasResource = other.hasResource;
        }
        other.binding = nullptr;
    }

    BoundResource& operator=(BoundResource&& other) {
        if (this == &other)
            return *this;
        close();
        binding = other.binding;
        if (binding) {
            std::unique_lock<std::mutex> lock(binding->m);
            parents = std::move(other.parents);
            value = std::move(other.value);
            data = std::move(other.data);
            gen = std::move(other.gen);
            del = std::move(other.del);
            hasResource = other.hasResource;
            binding->resource = this;
        }
        else {
            parents = std::move(other.parents);
            value = std::move(other.value);
            data = std::move(other.data);
            gen = std::move(other.gen);
            del = std::move(other.del);
            hasResource = other.hasResource;
        }
        other.binding = nullptr;
        return *this;
    }

    ~BoundResource() { close(); }

    void destroy(BoundResourceParent* parent) override final {
        drv::drv_assert(hasResource);
        for (BindingParentData* p : parents) {
            if (parent == p->parent)
                continue;
            std::unique_lock<std::mutex> lock(p->m);
            p->parent->unbind(binding);
        }
        del(data, std::move(value));
        parents.clear();
        hasResource = false;
    }

    T& bind(uint32_t parentCount, BoundResourceParent** parentResources) {
        if (!hasResource) {
            close();
            value = gen(data, parentCount, parentResources);
            for (uint32_t i = 0; i < parentCount; ++i)
                parents.push_back(parentResources->bind(binding));
            hasResource = true;
        }
        return get();
    }

    const T& get() const {
        drv::drv_assert(hasResource, "Resource doesn't exist");
        return value;
    }

    T& get() {
        drv::drv_assert(hasResource, "Resource doesn't exist");
        return value;
    }

 private:
    BindingData* binding = nullptr;
    std::vector<BindingParentData*> parents;
    T value;
    D data;
    Gen gen;
    Del del;
    bool hasResource = false;

    void close() {
        if (binding) {
            if (hasResource)
                destroy(nullptr);
            else
                drv::drv_assert(parents.size() == 0);
            delete binding;
            binding = nullptr;
        }
    }
};
