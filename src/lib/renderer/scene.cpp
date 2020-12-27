#include "scene.h"

void Scene::_render(const RenderContext& context) const {
    for (const ItemData& item : items) {
        item.renderable->render(context);
    }
}

void Scene::addItem(ItemData&& item) {
    items.push_back(std::move(item));
}

void Scene::clear() {
    items.clear();
}
