#pragma once

#include <memory>
#include <vector>

#include "renderable.h"

class Mesh;

class Scene : public RenderableInterface
{
 public:
    struct ItemData
    {
        std::unique_ptr<RenderableInterface> renderable;
    };

    void addItem(ItemData&& item);

    const std::vector<ItemData>& getItems() const { return items; }

    std::vector<std::string> getPrograms() const override { return {}; }
    std::vector<FloatOption> getOptions() override { return {}; }

    void clear();

 protected:
    void _render(const RenderContext& context) const override;

 private:
    std::vector<ItemData> items;
};
