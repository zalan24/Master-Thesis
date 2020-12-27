#include "renderable.h"

std::size_t RenderableInterface::itemCount = 0;

RenderableInterface::RenderableInterface() : name("item #" + std::to_string(itemCount++)) {
}

RenderableInterface::RenderableInterface(const std::string& n) : name(n) {
}

RenderableInterface::~RenderableInterface() {
}
