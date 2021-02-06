#pragma once

#include <string>

class ShaderDescriptor
{
 public:
    ShaderDescriptor();
    virtual ~ShaderDescriptor();

    ShaderDescriptor(const ShaderDescriptor&) = default;
    ShaderDescriptor& operator=(const ShaderDescriptor&) = default;
    ShaderDescriptor(ShaderDescriptor&&) = default;
    ShaderDescriptor& operator=(ShaderDescriptor&&) = default;

    virtual void setVariant(const std::string& vairantName, const std::string& value) = 0;
    virtual void setVariant(const std::string& vairantName, int value) = 0;

 private:
};
