#pragma once

#include <string>
#include <vector>

class ShaderDescriptor
{
 public:
    ShaderDescriptor();
    virtual ~ShaderDescriptor();

    ShaderDescriptor(const ShaderDescriptor&) = default;
    ShaderDescriptor& operator=(const ShaderDescriptor&) = default;
    ShaderDescriptor(ShaderDescriptor&&) = default;
    ShaderDescriptor& operator=(ShaderDescriptor&&) = default;

    virtual void setVariant(const std::string& variantName, const std::string& value) = 0;
    virtual void setVariant(const std::string& variantName, int value) = 0;
    virtual std::vector<std::string> getVariantParamNames() const = 0;
    virtual uint32_t getLocalVariantId() const = 0;

 private:
};
