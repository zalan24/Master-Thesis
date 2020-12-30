#include "loadmesh.h"

#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <glm/geometric.hpp>

#include <mesh.h>

static Mesh process(aiMesh* mesh) {
    Mesh ret;
    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        glm::vec3 pos = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        glm::vec3 normal{0, 0, 0};
        glm::vec3 color{1, 1, 1};
        glm::vec2 texture{0, 0};
        if (mesh->HasNormals())
            normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->HasTextureCoords(i))
            texture = glm::vec2(mesh->mTextureCoords[i]->x, mesh->mTextureCoords[i]->y);
        ret.addVertex({pos, normal, color, texture});
    }
    for (uint32_t i = 0; i < mesh->mNumFaces; ++i) {
        assert(mesh->mFaces->mNumIndices == 3);
        for (uint32_t j = 0; j < 3; ++j)
            if (mesh->mFaces[i].mIndices[j] >= mesh->mNumVertices) {
                throw std::runtime_error("Could not load mesh: Vertex index out of range");
            }
        unsigned int p1 = mesh->mFaces[i].mIndices[0];
        unsigned int p2 = mesh->mFaces[i].mIndices[1];
        unsigned int p3 = mesh->mFaces[i].mIndices[2];
        ret.addFace(p1, p2, p3);
    }
    return ret;
}

std::vector<Mesh> loadMesh(const std::string& filename) {
    std::vector<Mesh> ret;
    Assimp::Importer importer;
    const aiScene* scene =
      importer.ReadFile(filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate
                                    | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    if (!scene) {
        //DoTheErrorLogging(importer.GetErrorString());
        std::string vError = importer.GetErrorString();
        throw std::runtime_error("Could not load object from file: " + filename + " (" + vError
                                 + ")");
    }
    if (scene->HasMeshes()) {
        for (uint32_t i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh* m = scene->mMeshes[i];
            //vMesh->mNumVertices
            Mesh mesh = process(m);
            ret.push_back(std::move(mesh));
        }
    }
    return std::move(ret);
}

Mesh loadMeshCube() {
    Mesh ret;
    for (int ind : {0, 1, 2}) {
        for (int sgn : {-1, 1}) {
            glm::vec3 dir{0, 0, 0};
            dir[ind] = sgn;
            glm::vec3 side{0, 0, 0};
            side[(ind + 1) % 3] = sgn;
            glm::vec3 up = glm::cross(dir, side);
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