#include "renderer.h"

#include <engine.h>

#include <shader_obj_test.h>

Renderer::Renderer(Engine* _engine) : engine(_engine) {
    shader_obj_test testShader(engine->getDevice(), *engine->getShaderBin());
    shader_obj_test::Descriptor descriptor;
}

Renderer::~Renderer() {
}
