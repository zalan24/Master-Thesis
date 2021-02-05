#pragma once

class IRenderer
{
 public:
    IRenderer() = default;
    virtual ~IRenderer();

    IRenderer(const IRenderer&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;

 private:
};
