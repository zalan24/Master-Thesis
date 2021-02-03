#include "shaderbin.h"

#include <stdexcept>

void ShaderBin::read(std::istream& in) {
    throw std::runtime_error("Cannot read yet");
}

void ShaderBin::write(std::ostream& out) const {
    throw std::runtime_error("Cannot write yet");
}

void ShaderBin::addShader(const std::string& name, ShaderData&& shader) {
    if (shaders.find(name) != shaders.end())
        throw std::runtime_error("A shader already exists with the given name: " + name);
    shaders[name] = std::move(shader);
}
