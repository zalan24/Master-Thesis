#pragma once

#include <memory>

#include <irenderer.h>
#include <shaderobject.h>

class Engine;

class Renderer final : public IRenderer
{
 public:
    Renderer(Engine* engine);
    ~Renderer() override;

 private:
    Engine* engine;
};
