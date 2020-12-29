#pragma once

#include <serializable.h>

#include "spheremesh.h"

using uint = unsigned int;

class Sampler
  : public RenderableInterface
  , public ISerializable
{
 public:
    using VertexIndex = uint32_t;
    struct BuildData
    {
        uint sphericalCoordinates;
        uint cubeMap;
        uint tetrahedron;
        uint octahedron;
    };

    struct Sample
    {
        glm::vec3 pos;
        glm::vec2 gridPos;
        glm::vec3 polyPos;
        glm::vec3 spherePos;
        glm::vec2 tex;
        glm::vec3 sideColor;
    };

    Sampler();

    void build(const BuildData& data, const SphereMesh* sphereMesh);
    BuildData getBuildData() const { return config; }

    enum class MapMode
    {
        NONE,
        SPHERICAL_COORDINATES,
        CUBE_MAP,
        TETRAHEDRON,
        OCTAHEDRON
    };

    void display(MapMode mode);

    std::vector<std::string> getPrograms() const override { return {shaderProgram}; }
    std::vector<FloatOption> getOptions() override;
    std::vector<BoolOption> getBoolOptions() override;

    bool isBuilt() const;

    void write(std::ostream& out) const override;
    void read(std::istream& in) override;

 protected:
    void gatherEntries(std::vector<ISerializable::Entry>& entries) const override {}
    void _render(const RenderContext& context) const override;

 private:
    BuildData config;

    AttributeBinder attributeBinder;

    std::string shaderProgram;

    Buffer<Sample> glBuffer;
    Buffer<VertexIndex> glIndices;
    glm::mat4 modelTransform;

    std::vector<Sample> sphericalCoordinatesSamples;
    std::vector<Sample> cubeMapSamples;
    std::vector<Sample> tetrahedronSamples;
    std::vector<Sample> octahedronSamples;
    std::vector<VertexIndex> indices;
    int numData = 1;
    float interpolation = 1;
    bool built = false;
    MapMode displayMode = MapMode::NONE;

    std::size_t index(uint x, uint y, uint width) const { return y * width + x; }

    void sampleSC(const BuildData& data, const SphereMesh* sphereMesh, std::vector<Sample>& vec,
                  uint size);
    void sampleOH(const BuildData& data, const SphereMesh* sphereMesh, std::vector<Sample>& vec,
                  uint size);

    bool displayCoords = false;
};
