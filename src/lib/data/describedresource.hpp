#pragma once

#include <serializable.h>

#include "resourcepool.h"

template <typename P, typename D = typename P::ResourceDescriptor>
class DescribedResource final : public ISerializable
{
 public:
    DescribedResource(const P* _provider, D&& _descriptor)
      : provider(_provider),
        descriptor(std::move(_descriptor)),
        resource(provider->getResource(descriptor)) {}

    const GenericResourcePool::ResourceRef& getRes() const { return resource; }

 protected:
    void gatherEntries(std::vector<Entry>& entries) const override {
        REGISTER_OBJECT(descriptor, entries);
    }

 private:
    const P* provider;
    D descriptor;
    GenericResourcePool::ResourceRef resource;
};
