#include "shaderdescriptorcollection.h"

ShaderDescriptorCollection::~ShaderDescriptorCollection() {
}

ShaderDescriptor* ShaderDescriptorCollection::getDescriptorByName(const std::string& name) {
    auto itr = nameToDesc.find(name);
    if (itr == nameToDesc.end())
        throw std::runtime_error("Could not find descriptor: " + name);
    return getDesc(itr->second);
}

const ShaderDescriptor* ShaderDescriptorCollection::getDescriptorByName(
  const std::string& name) const {
    auto itr = nameToDesc.find(name);
    if (itr == nameToDesc.end())
        throw std::runtime_error("Could not find descriptor: " + name);
    return getDesc(itr->second);
}

void ShaderDescriptorCollection::setVariant(const std::string& variantName,
                                            const std::string& value) {
    auto itr = variantParamToDesc.find(variantName);
    if (itr == variantParamToDesc.end())
        throw std::runtime_error("Could not find variant param in shader descriptor: "
                                 + variantName);
    getDesc(itr->second)->setVariant(variantName, value);
}

void ShaderDescriptorCollection::setVariant(const std::string& variantName, int value) {
    auto itr = variantParamToDesc.find(variantName);
    if (itr == variantParamToDesc.end())
        throw std::runtime_error("Could not find variant param in shader descriptor: "
                                 + variantName);
    getDesc(itr->second)->setVariant(variantName, value);
}

std::vector<std::string> ShaderDescriptorCollection::getVariantParamNames() const {
    return variantParams;
}
