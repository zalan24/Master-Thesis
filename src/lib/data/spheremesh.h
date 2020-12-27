#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "intersections.h"
#include "mesh.h"
#include "serializable.h"
#include "spheretree.hpp"

class SphereMesh
  : public RenderableInterface
  , public ISerializable
{
 public:
    SphereMesh(const Mesh& mesh, const std::string& shaderProgram = "spheremesh");
    SphereMesh(const std::string& shaderProgram = "spheremesh");
    ~SphereMesh();

    SphereMesh(const SphereMesh&) = delete;
    SphereMesh& operator=(const SphereMesh&) = delete;

    void write(std::ostream& out) const override;
    void read(std::istream& in) override;

    std::vector<std::string> getPrograms() const override;
    std::vector<FloatOption> getOptions() override;

    enum class Status
    {
        UNINITIALIZED,
        GRAPH_READY,
        REPAIRED,
        MATRIX_READY,
        PREPROCESSED,
        PROCESSED,
        VERTEX_DATA_READY,
        UPLOADED
    };
    struct VertexData
    {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
        glm::vec2 texcoord;
        glm::vec3 new_position;
    };

    // these will be the initial values of the non-static members
    // these can be set from the gui
    static float sMatrixAlpha_option;
    static float sShoothenBias_option;
    static int sPreprocessCount_option;
    static int sProcessMax_option;

    void rebuild();
    void refine();
    void terminateBuild();
    int getProgress() const { return progress; }

    Status getStatus() const { return status; }

    VertexData getSample(const glm::vec2& sc) const;

 protected:
    void _render(const RenderContext& context) const override;

 private:
    static const unsigned int MAX_THREADS = 3;
    using VertexIndex = uint32_t;
    using FaceIndex = uint32_t;
    struct EdgeData
    {
        float length;
    };
    struct NodeData
    {
        std::map<VertexIndex, EdgeData> e;
        std::set<FaceIndex> f;
        glm::vec3 position;
        glm::vec3 originalPosition;
        glm::vec3 color;
        glm::vec2 texcoord;
    };
    std::vector<std::unique_ptr<NodeData>> nodes;
    std::vector<VertexIndex> remap;
    std::vector<std::map<VertexIndex, float>> A;

    std::string shaderProgram;  // these are not saved to cache
    std::string outlinerProgram;

    std::vector<VertexData> vertices;
    std::vector<VertexIndex> index;
    Buffer<VertexData> glBuffer;
    Buffer<VertexIndex> glIndices;
    AttributeBinder attributeBinder;
    AttributeBinder outlineAttributeBinder;
    glm::mat4 modelTransform;

    mutable Status status = Status::UNINITIALIZED;
    mutable std::unique_ptr<std::thread> thread;
    mutable std::vector<glm::vec3> temp_positions;

    SphereTree<VertexIndex> bvTree;

    void bindAttributes();

    void buildGraph(const Mesh& mesh);
    void repair(const Mesh& mesh);
    void preprocess();
    void process(int repeat);
    void createVertexData();
    void uploadData() const;
    void compile(const Mesh& mesh);

    void project(const glm::vec3& from, float size = 1);
    void smoothen(float extra);
    void smoothen(float extra, int repeat);
    void smartProcess();
    void createMatrix(float alpha);
    void adjust();
    bool isOverlapping(bool paint = false);
    std::vector<std::tuple<VertexIndex, VertexIndex>> getHoleEdges(VertexIndex i,
                                                                   VertexIndex j) const;

    void addTriangle(Mesh::VertexIndex ind, Mesh::VertexIndex n1, Mesh::VertexIndex n2,
                     const glm::vec3& indPos, const glm::vec3& n1Pos, const glm::vec3& n2Pos);

    Triangle getTriangle(std::size_t ind) const;

    float interpolation = 1;
    float matrixAlpha = 0;
    float shoothenBias = 0.25;
    unsigned int preprocessCount = 0;
    unsigned int processMax = 0;

    int progress = 0;
};
