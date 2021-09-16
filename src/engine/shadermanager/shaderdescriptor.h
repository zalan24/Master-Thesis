#pragma once

#include <string>
#include <vector>

class ShaderDescriptorReg
{
 public:
 protected:
    ~ShaderDescriptorReg() {}
};
class ShaderDescriptor
{
 public:
    using DataVersionNumber = uint64_t;
    explicit ShaderDescriptor(std::string name);
    virtual ~ShaderDescriptor();

    ShaderDescriptor(const ShaderDescriptor&) = default;
    ShaderDescriptor& operator=(const ShaderDescriptor&) = default;
    ShaderDescriptor(ShaderDescriptor&&) = default;
    ShaderDescriptor& operator=(ShaderDescriptor&&) = default;

    virtual void setVariant(const std::string& variantName, const std::string& value) = 0;
    virtual void setVariant(const std::string& variantName, int value) = 0;
    virtual std::vector<std::string> getVariantParamNames() const = 0;
    virtual uint32_t getLocalVariantId() const = 0;
    virtual uint32_t getPushConstStructIdGraphics() const = 0;
    virtual uint32_t getPushConstStructIdCompute() const = 0;
    virtual void pushGraphicsConsts(void* dst) const = 0;

    DataVersionNumber getPushConstsVersionNumber() const { return pushConstVersionNumber; }

    const std::string& getName() const { return name; }

    virtual const ShaderDescriptorReg* getReg() const = 0;

 protected:
    //  TODO;  // don't invalidate when a param is changed, that's not used in the variant???
    //  TODO;  // invalidate when variant is changed and the used params struct changes
    void invalidatePushConsts() { pushConstVersionNumber++; }

 private:
    std::string name;
    DataVersionNumber pushConstVersionNumber = 0;
};
