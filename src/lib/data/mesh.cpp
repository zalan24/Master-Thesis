#include "mesh.h"

#include <glad/glad.h>

#include "util.hpp"

#include <algorithm>

Mesh::Mesh(const std::string& s)
  : shaderProgram(s),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER),
    modelTransform(1) {
    attributeBinder.addAttribute(&VertexData::position, "vPos");
    attributeBinder.addAttribute(&VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&VertexData::color, "vCol");
    // attributeBinder.addAttribute(&VertexData::texcoord, "vTex");
}

void Mesh::setShader(const std::string& s) {
    built = false;
    shaderProgram = s;
}

Mesh::VertexIndex Mesh::addVertex(const VertexData& vert) {
    built = false;
    VertexIndex ret = safeCast<VertexIndex>(vertices.size());
    vertices.push_back(vert);
    return ret;
}

void Mesh::addFace() {
    built = false;
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind, ind + 1, ind + 2);
}

void Mesh::addFaceRev() {
    built = false;
    VertexIndex ind =
      safeCast<VertexIndex>(assertReturn(vertices.size(), [](auto val) { return val >= 3; }) - 3);
    addFace(ind + 2, ind, ind + 1);
}

void Mesh::addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3) {
    index.push_back(p1);
    index.push_back(p2);
    index.push_back(p3);
}

void Mesh::_render(const RenderContext& context) const {
    if (!built)
        throw std::runtime_error("Mesh was renderer without building it first");
    context.shaderManager->useProgram(shaderProgram);
    glBuffer.bind();
    glIndices.bind();
    attributeBinder.bind(*context.shaderManager);
    context.shaderManager->setUniform("PVM", context.pv * modelTransform);
    context.shaderManager->setUniform("model", modelTransform);
    context.shaderManager->setUniform("lightColor", context.lightColor);
    context.shaderManager->setUniform("lightDir", context.lightDir);
    context.shaderManager->setUniform("ambientColor", context.ambientColor);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, index.size(), GL_UNSIGNED_INT, 0);
}

void Mesh::build() {
    if (built)
        return;
    glBuffer.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(index, GL_STATIC_DRAW);
    built = true;
}

std::vector<std::string> Mesh::getPrograms() const {
    return {shaderProgram};
}

std::vector<FloatOption> Mesh::getOptions() {
    return {};
}

void Mesh::normalize() {
    glm::vec3 G{0, 0, 0};
    float maxDist = 0;
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        G += vertices[i].position;
    }
    G /= vertices.size();
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        maxDist = std::max(maxDist, glm::distance(G, vertices[i].position));
    }
    for (VertexIndex i = 0; i < vertices.size(); ++i) {
        vertices[i].position = (vertices[i].position - G) / maxDist;
    }
}
