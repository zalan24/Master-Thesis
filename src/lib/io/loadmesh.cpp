#include "loadmesh.h"

#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <glm/geometric.hpp>

#include <mesh.h>
#include <util.hpp>

static void process(const aiMesh* mesh, Mesh& m) {
    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        assert(mesh->HasPositions());
        glm::vec3 pos = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        glm::vec3 normal{0, 0, 0};
        glm::vec3 color{1, 1, 1};
        glm::vec2 texture{0, 0};
        if (mesh->HasNormals())
            normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->HasTextureCoords(i))
            texture = glm::vec2(mesh->mTextureCoords[i]->x, mesh->mTextureCoords[i]->y);
        m.addVertex({pos, normal, color, texture});
    }
    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        assert(mesh->mFaces->mNumIndices == 3);
        for (uint32_t j = 0; j < 3; ++j) {
            if (mesh->mFaces[i].mIndices[j] >= mesh->mNumVertices) {
                throw std::runtime_error("Could not load mesh: Vertex index out of range");
            }
        }
        unsigned int p1 = mesh->mFaces[i].mIndices[0];
        unsigned int p2 = mesh->mFaces[i].mIndices[1];
        unsigned int p3 = mesh->mFaces[i].mIndices[2];
        m.addFace(p1, p2, p3);
    }
}

static Mesh process(const aiScene* scene, const aiNode* node) {
    assert(node != nullptr);
    Mesh ret(node->mName.length == 0 ? "" : std::string(node->mName.data, node->mName.length));
    glm::mat4 tm;
    tm[0][0] = node->mTransformation.a1;
    tm[1][0] = node->mTransformation.a2;
    tm[2][0] = node->mTransformation.a3;
    tm[3][0] = node->mTransformation.a4;
    tm[0][1] = node->mTransformation.b1;
    tm[1][1] = node->mTransformation.b2;
    tm[2][1] = node->mTransformation.b3;
    tm[3][1] = node->mTransformation.b4;
    tm[0][2] = node->mTransformation.c1;
    tm[1][2] = node->mTransformation.c2;
    tm[2][2] = node->mTransformation.c3;
    tm[3][2] = node->mTransformation.c4;
    tm[0][3] = node->mTransformation.d1;
    tm[1][3] = node->mTransformation.d2;
    tm[2][3] = node->mTransformation.d3;
    tm[3][3] = node->mTransformation.d4;
    ret.setNodeTm(tm);
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        if (i == 0)
            process(scene->mMeshes[node->mMeshes[i]], ret);
        else {
            Mesh m;
            process(scene->mMeshes[node->mMeshes[i]], m);
            ret.addChild(std::move(m));
        }
    }
    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        ret.addChild(process(scene, node->mChildren[i]));
    return ret;
}

Mesh loadMesh(const std::string& filename) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
                  | aiProcess_SortByPType | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords
                  | aiProcess_EmbedTextures);
    if (!scene) {
        std::string vError = importer.GetErrorString();
        throw std::runtime_error("Could not load object from file: " + filename + " (" + vError
                                 + ")");
    }
    return process(scene, scene->mRootNode);
}

Mesh createCube(float size) {
    Mesh ret;
    for (int ind : {0, 1, 2}) {
        for (int sgn : {-1, 1}) {
            glm::vec3 dir{0, 0, 0};
            dir[ind] = sgn;
            glm::vec3 side{0, 0, 0};
            side[(ind + 1) % 3] = sgn;
            glm::vec3 up = glm::cross(dir, side) * size;
            side *= size;
            dir *= size;
            glm::vec3 normal = glm::normalize(dir);
            ret.addVertex(
              Mesh::VertexData{dir - side - up, normal, glm::vec3{1, 1, 1}, glm::vec2{0, 0}});
            ret.addVertex(
              Mesh::VertexData{dir - side + up, normal, glm::vec3{1, 0, 0}, glm::vec2{0, 0}});
            ret.addVertex(
              Mesh::VertexData{dir + side - up, normal, glm::vec3{0, 1, 0}, glm::vec2{0, 0}});
            ret.addFace();
            ret.addVertex(
              Mesh::VertexData{dir + side + up, normal, glm::vec3{0, 0, 1}, glm::vec2{0, 0}});
            ret.addFaceRev();
        }
    }
    return ret;
}

Mesh createPlane(const glm::vec3& origin, glm::vec3 normal, float size) {
    normal = glm::normalize(normal);
    glm::vec3 up =
      std::abs(normal.y) > std::abs(normal.z) ? glm::vec3{0, 0, 1} : glm::vec3{0, 1, 0};
    up -= normal * dot(up, normal);
    glm::vec3 side = glm::cross(normal, up);
    up *= size;
    side *= size;
    Mesh ret;
    ret.addVertex(
      Mesh::VertexData{origin - side - up, normal, glm::vec3{1, 1, 1}, glm::vec2{0, 0}});
    ret.addVertex(
      Mesh::VertexData{origin - side + up, normal, glm::vec3{1, 0, 0}, glm::vec2{0, 1}});
    ret.addVertex(
      Mesh::VertexData{origin + side - up, normal, glm::vec3{0, 1, 0}, glm::vec2{1, 0}});
    ret.addFace();
    ret.addVertex(
      Mesh::VertexData{origin + side + up, normal, glm::vec3{0, 0, 1}, glm::vec2{1, 1}});
    ret.addFaceRev();
    return ret;
}

Mesh createSphere(size_t resX, size_t resY, float size) {
    Mesh ret;
    for (size_t y = 0; y < resY; ++y) {
        float theta = static_cast<float>(y) / (resY - 1);
        float fy = std::cos(theta * M_PI);
        float fxz = std::sin(theta * M_PI);
        for (size_t x = 0; x < resX; ++x) {
            float phi = static_cast<float>(x) / resX;
            float fx = std::cos(phi * 2 * M_PI) * fxz;
            float fz = std::sin(phi * 2 * M_PI) * fxz;
            glm::vec3 normal{fx, fy, fz};
            glm::vec3 pos = normal * size;
            glm::vec2 texcoord{phi, theta};
            ret.addVertex(Mesh::VertexData{pos, normal, glm::vec3{1, 0, 0}, texcoord});
        }
    }
    for (size_t y = 0; y < resY - 1; ++y) {
        for (size_t x = 0; x < resX; ++x) {
            Mesh::VertexIndex v00 = x + y * resX;
            Mesh::VertexIndex v01 = ((x + 1) % resX) + y * resX;
            Mesh::VertexIndex v10 = x + ((y + 1) % resY) * resX;
            Mesh::VertexIndex v11 = ((x + 1) % resX) + ((y + 1) % resY) * resX;
            ret.addFace(v00, v01, v10);
            ret.addFace(v11, v01, v10);
        }
    }
    return ret;
}
