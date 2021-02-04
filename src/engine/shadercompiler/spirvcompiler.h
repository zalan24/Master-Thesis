#pragma once

#include <shaderbin.h>

class Compiler
{
 public:
    Compiler();
    ~Compiler();

    Compiler(const Compiler&) = delete;
    Compiler& operator=(const Compiler&) = delete;

    std::vector<uint32_t> GLSLtoSPV(const ShaderBin::Stage stage, const char* pShader) const;

 private:
};
