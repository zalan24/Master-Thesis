#pragma once

#include <string>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "buffer.hpp"
#include "renderable.h"
#include "shadermanager.h"

class Mesh : public RenderableInterface
{
 public:
    using VertexIndex = uint32_t;
    struct VertexData
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec2 texcoord;
    };

    Mesh(const std::string& shaderProgram = "dummy");
    ~Mesh() noexcept {}

    void setShader(const std::string& shaderName);

    VertexIndex addVertex(const VertexData& vert);
    void addFace();     // adds last three vertices
    void addFaceRev();  // adds last three vertices flipped
    void addFace(VertexIndex p1, VertexIndex p2, VertexIndex p3);

    virtual void build();

    std::vector<std::string> getPrograms() const override;

    std::vector<FloatOption> getOptions() override;

    void normalize();

 protected:
    friend class SphereMesh;

    std::string shaderProgram;

    std::vector<VertexData> vertices;
    std::vector<VertexIndex> index;
    Buffer<VertexData> glBuffer;
    Buffer<VertexIndex> glIndices;
    AttributeBinder attributeBinder;
    glm::mat4 modelTransform;

    bool built = false;

    void _render(const RenderContext& context) const override;
};
