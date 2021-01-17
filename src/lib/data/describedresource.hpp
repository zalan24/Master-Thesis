#pragma once

#include <serializable.h>

#include "resourcepool.h"

template <typename P, typename D = typename P::ResourceDescriptor>
class DescribedResource final : public ISerializable
{
 public:
    DescribedResource() {}
    DescribedResource(D&& _descriptor) : descriptor(std::move(_descriptor)) {
        resource = P::getSingleton()->getResource(descriptor);
    }

    const GenericResourcePool::ResourceRef& getRes() const { return resource; }

    void writeJson(json& out) const override final { WRITE_OBJECT(descriptor, out); }
    void readJson(const json& in) override final {
        READ_OBJECT(descriptor, in);
        resource = P::getSingleton()->getResource(descriptor);
    }

 private:
    D descriptor;
    GenericResourcePool::ResourceRef resource;
};
