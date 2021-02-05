#pragma once

#include <memory>

#include <irenderer.h>
#include <shaderobject.h>

class Renderer final : public IRenderer
{
 public:
    Renderer();
    ~Renderer() override;

 private:
};
