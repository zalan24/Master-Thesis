#include "textureprovider.h"

TextureProvider* TextureProvider::instance = nullptr;

TextureProvider::TextureProvider() {
    assert(instance == nullptr);
    instance = this;
}

TextureProvider::~TextureProvider() {
    assert(instance == this);
    instance = nullptr;
}

GenericResourcePool::ResourceRef TextureProvider::createResource(const RGBA& color) const {
    Texture<RGBA> tex(1, 1, 1);
    tex.set(color, 0, 0, 0);
    return createResource(std::move(tex));
}

TextureProvider::ResourceDescriptor::ResourceDescriptor(const glm::vec4& _value)
  : value(_value), filename("") {
}

TextureProvider::ResourceDescriptor::ResourceDescriptor(const std::string& _filename)
  : filename(_filename) {
}

void TextureProvider::ResourceDescriptor::writeJson(json& out) const {
    const float* color = &value.x;
    WRITE_OBJECTS(color, 4, out);
    WRITE_OBJECT(filename, out);
}

void TextureProvider::ResourceDescriptor::readJson(const json& in) {
    float* color = &value.x;
    READ_OBJECTS(color, 4, in);
    READ_OBJECT(filename, in);
}
