#pragma once

#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "shaderdescriptor.h"

class ShaderDescriptorCollection : public ShaderDescriptor
{
 public:
    ~ShaderDescriptorCollection() override;

    void setVariant(const std::string& variantName, const std::string& value) override final;
    void setVariant(const std::string& variantName, int value) override final;
    std::vector<std::string> getVariantParamNames() const override final;

    ShaderDescriptor* getDescriptorByName(const std::string& name);
    const ShaderDescriptor* getDescriptorByName(const std::string& name) const;

    template <typename D>
    ShaderDescriptor* getDescriptorByType() {
        std::type_index index = std::type_index(typeid(D));
        auto itr = typeToDesc.find(index);
        if (itr == typeToDesc.end())
            throw std::runtime_error("Could not find descriptor with type: "
                                     + std::string(index.name()));
        return getDesc(itr->second);
    }

    template <typename D>
    const ShaderDescriptor* getDescriptorByType() const {
        std::type_index index = std::type_index(typeid(D));
        auto itr = typeToDesc.find(index);
        if (itr == typeToDesc.end())
            throw std::runtime_error("Could not find descriptor with type: "
                                     + std::string(index.name()));
        return getDesc(itr->second);
    }

 protected:
    using DescId = uint32_t;

    virtual ShaderDescriptor* getDesc(const DescId& id) = 0;
    virtual const ShaderDescriptor* getDesc(const DescId& id) const = 0;

    template <typename D>
    void addDescriptor(const std::string& name, const DescId& id, const D& descriptor) {
        std::type_index index = std::type_index(typeid(D));
        if (nameToDesc.find(name) != nameToDesc.end())
            throw std::runtime_error("A shader descriptor is already added with this name: "
                                     + name);
        if (typeToDesc.find(index) != typeToDesc.end())
            throw std::runtime_error("A shader descriptor is already added with this type: "
                                     + std::string(index.name()));
        for (const std::string& paramName : descriptor.getVariantParamNames())
            if (auto itr = variantParamToDesc.find(paramName); itr != variantParamToDesc.end())
                throw std::runtime_error("Variant param name must be unique (" + paramName + ")");
        for (const std::string& paramName : descriptor.getVariantParamNames()) {
            variantParamToDesc[paramName] = id;
            variantParams.push_back(paramName);
        }
        nameToDesc[name] = id;
        typeToDesc[index] = id;
    }

 private:
    std::vector<std::string> variantParams;
    std::unordered_map<std::string, uint32_t> nameToDesc;
    std::unordered_map<std::string, uint32_t> variantParamToDesc;
    std::unordered_map<std::type_index, uint32_t> typeToDesc;
};
