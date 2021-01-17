#include "textureprovider.h"

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
    // TODO support vec4
    // REGISTER_ENTRY(value, entries);
    WRITE_OBJECT(filename, out);
}

void TextureProvider::ResourceDescriptor::readJson(const json& in) {
    // TODO support vec4
    // REGISTER_ENTRY(value, entries);
    READ_OBJECT(filename, in);
}
