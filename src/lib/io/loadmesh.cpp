#include "loadmesh.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#define GLM_FORCE_RADIANS
#define GLM_LEFT_HAND
#include <glm/geometric.hpp>

#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <material.h>
#include <mesh.h>
#include <textureprovider.h>
#include <util.hpp>

#include "loadimage.h"

// Thanks to http://www.ogldev.org/www/tutorial38/tutorial38.html for the Skeletal animations tutorial

static glm::mat4 convert_matrix(const aiMatrix4x4& mat) {
    glm::mat4 tm;
    tm[0][0] = mat.a1;
    tm[1][0] = mat.a2;
    tm[2][0] = mat.a3;
    tm[3][0] = mat.a4;
    tm[0][1] = mat.b1;
    tm[1][1] = mat.b2;
    tm[2][1] = mat.b3;
    tm[3][1] = mat.b4;
    tm[0][2] = mat.c1;
    tm[1][2] = mat.c2;
    tm[2][2] = mat.c3;
    tm[3][2] = mat.c4;
    tm[0][3] = mat.d1;
    tm[1][3] = mat.d2;
    tm[2][3] = mat.d3;
    tm[3][3] = mat.d4;
    return tm;
}

static glm::mat4 get_bone_transform(const Mesh::Skeleton* skeleton, Mesh::BoneIndex boneId) {
    const Mesh::Bone& b = skeleton->getBone(boneId);
    if (b.parent != Mesh::INVALID_BONE && b.parent != boneId)
        return get_bone_transform(skeleton, b.parent) * b.localTm;
    return b.localTm;
}

static Mesh::Segment process(const aiMesh* mesh, const std::vector<Mesh::MaterialIndex>& materials,
                             Mesh::BoneIndex& boneId, const Mesh::Skeleton* skeleton,
                             const glm::vec3& default_color) {
    assert(mesh->GetNumUVChannels() <= 1);
    Mesh::Segment segment;
    // mesh->mAnimMeshes[0]->
    std::map<uint32_t, std::vector<std::pair<uint32_t, float>>> vertexBoneWeights;
    segment.boneOffsets = std::vector<glm::mat4>(skeleton->getBoneCount(), glm::mat4(1.f));
    if (mesh->HasBones()) {
        for (unsigned int i = 0; i < mesh->mNumBones; ++i) {
            std::string name(mesh->mBones[i]->mName.C_Str());
            Mesh::BoneIndex id = skeleton->getBoneId(name);
            assert(id != Mesh::INVALID_BONE);
            segment.boneOffsets[id] = convert_matrix(mesh->mBones[i]->mOffsetMatrix);
            for (unsigned int j = 0; j < mesh->mBones[i]->mNumWeights; ++j)
                vertexBoneWeights[mesh->mBones[i]->mWeights[j].mVertexId].emplace_back(
                  id, mesh->mBones[i]->mWeights[j].mWeight);
        }
        for (auto& itr : vertexBoneWeights) {
            // assert(itr.second.size() <= Mesh::MAX_BONES);  // TODO not sure if this would work
            if (itr.second.size() > Mesh::MAX_BONES) {
                std::sort(
                  itr.second.begin(), itr.second.end(),
                  [](const std::pair<uint32_t, float>& lhs, const std::pair<uint32_t, float>& rhs) {
                      return lhs.second > rhs.second;
                  });
                itr.second.resize(Mesh::MAX_BONES);
            }
        }
    }
    if (mesh->mMaterialIndex <= materials.size())
        segment.mat = materials[mesh->mMaterialIndex];
    else
        assert(materials.size() == 0);
    for (uint32_t i = 0; i < mesh->mNumVertices; ++i) {
        assert(mesh->HasPositions());
        glm::vec3 pos = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        glm::vec3 normal{0, 0, 0};
        glm::vec3 color = default_color;
        glm::vec2 texture{0, 0};
        glm::ivec4 boneIds(boneId, boneId, boneId, boneId);
        glm::vec4 boneWeights(1, 0, 0, 0);
        auto boneItr = vertexBoneWeights.find(i);
        if (boneItr != vertexBoneWeights.end() && boneItr->second.size() > 0) {
            for (size_t j = 0; j < boneItr->second.size(); ++j) {
                const auto& [id, weight] = boneItr->second[j];
                boneIds[j] = static_cast<float>(id);
                boneWeights[j] = weight;
            }
            boneWeights /= boneWeights.x + boneWeights.y + boneWeights.z + boneWeights.w;
        }
        if (mesh->HasNormals())
            normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->HasTextureCoords(0))
            texture = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        if (mesh->HasVertexColors(0)) {
            assert(false);  // TODO checks this (never been tried)
            color.r = mesh->mColors[0][i].r;
            color.g = mesh->mColors[0][i].g;
            color.b = mesh->mColors[0][i].b;
        }
        segment.addVertex(Mesh::VertexData(pos, normal, color, texture, boneIds, boneWeights));
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
        segment.addFace(p1, p2, p3);
    }
    return segment;
}

static Texture<RGBA> load_texture(const aiTexture* tex) {
    if (tex->mHeight == 0) {
        // compressed
        return load_image<RGBA>(static_cast<const void*>(tex->pcData), tex->mWidth);
    }
    else {
        Texture<RGBA> ret(tex->mHeight, tex->mHeight);
        for (unsigned int y = 0; y < tex->mHeight; ++y) {
            for (unsigned int x = 0; x < tex->mWidth; ++x) {
                RGBA pixel;
                const aiTexel& texel = tex->pcData[y * tex->mWidth + x];
                pixel.r = texel.r;
                pixel.g = texel.g;
                pixel.b = texel.b;
                pixel.a = texel.a;
                ret.set(std::move(pixel), x, y);
            }
        }
        assert(false);  // TODO checks this (never been tried)
        return ret;
    }
}

static void process(const aiScene* scene, const aiNode* node,
                    std::vector<Mesh::BoneIndex>& meshBones, Mesh::Skeleton* skeleton,
                    Mesh::BoneIndex parentBone) {
    assert(node != nullptr);
    Mesh::Bone bone;
    bone.parent = parentBone;
    bone.localTm = convert_matrix(node->mTransformation);
    Mesh::BoneIndex boneId = skeleton->addBone(std::move(bone));
    if (node->mName.length > 0)
        skeleton->registerBone(boneId, std::string(node->mName.C_Str()));
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        // Mesh bone should only be set once per segment
        assert(meshBones[node->mMeshes[i]] == skeleton->getRoot());
        meshBones[node->mMeshes[i]] = boneId;
    }
    for (unsigned int i = 0; i < node->mNumChildren; ++i)
        process(scene, node->mChildren[i], meshBones, skeleton, boneId);
}

Mesh load_mesh(const std::string& filename, const TextureProvider* texProvider,
               const glm::vec3& default_color) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(
      filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
                  | aiProcess_SortByPType | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords
                  | aiProcess_TransformUVCoords | aiProcess_EmbedTextures);
    if (!scene) {
        std::string vError = importer.GetErrorString();
        throw std::runtime_error("Could not load object from file: " + filename + " (" + vError
                                 + ")");
    }
    Mesh ret;
    std::vector<Mesh::MaterialIndex> materials;
    std::map<std::string, GenericResourcePool::ResourceRef> textures;
    if (scene->HasTextures()) {
        for (unsigned int i = 0; i < scene->mNumTextures; ++i) {
            std::string imgFile(scene->GetShortFilename(scene->mTextures[i]->mFilename.C_Str()));
            const std::string id = "*" + std::to_string(i);
            textures[id] = texProvider->createResource(load_texture(scene->mTextures[i]));
            if (imgFile != "")
                textures[imgFile] = textures[id];
        }
    }
    if (scene->HasMaterials()) {
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
            const auto matPtr = scene->mMaterials[i];
            Mesh::MaterialIndex mat;
            aiString path;
            if (matPtr->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS) {
                std::string imgFile(scene->GetShortFilename(path.C_Str()));
                auto itr = textures.find(imgFile);
                if (itr != textures.end()) {
                    mat = ret.addMaterial(Material(itr->second));
                }
                else {
                    TextureProvider::ResourceDescriptor matDesc{
                      std::string(path.data, path.length)};
                    mat = ret.addMaterial(
                      Material(Material::DiffuseRes(texProvider, std::move(matDesc))));
                    assert(false);  // TODO checks this (never been tried)
                }
            }
            else {
                aiColor3D albedo;
                float opacity = 1;
                matPtr->Get(AI_MATKEY_COLOR_DIFFUSE, albedo);
                matPtr->Get(AI_MATKEY_OPACITY, opacity);
                glm::vec4 albedo_alpha{albedo.r, albedo.g, albedo.b, opacity};
                TextureProvider::ResourceDescriptor matDesc{albedo_alpha};
                mat =
                  ret.addMaterial(Material(Material::DiffuseRes(texProvider, std::move(matDesc))));
            }
            materials.push_back(std::move(mat));
        }
    }
    std::vector<Mesh::BoneIndex> meshBones(scene->mNumMeshes, ret.getSkeleton()->getRoot());
    process(scene, scene->mRootNode, meshBones, ret.getSkeleton(), ret.getSkeleton()->getRoot());
    std::vector<Mesh::SegmentIndex> segments;
    if (scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            Mesh::Segment segment =
              process(scene->mMeshes[i], materials, meshBones[i], ret.getSkeleton(), default_color);
            segments.push_back(ret.addSegment(std::move(segment)));
        }
    }
    // TODO animations
    // scene->mAnimations[0]->mChannels[0]->;
    // scene->mAnimations[0]->mMeshChannels[0]->;
    ret.sortSegments();
    return ret;
}

Mesh create_cube(float size, const glm::vec3& color) {
    Mesh ret;
    Mesh::Segment segment;
    for (int ind : {0, 1, 2}) {
        for (int sgn : {-1, 1}) {
            glm::vec3 dir{0, 0, 0};
            dir[ind] = static_cast<float>(sgn);
            glm::vec3 side{0, 0, 0};
            side[(ind + 1) % 3] = static_cast<float>(sgn);
            glm::vec3 up = glm::cross(dir, side) * size;
            side *= size;
            dir *= size;
            glm::vec3 normal = glm::normalize(dir);
            segment.addVertex(Mesh::VertexData(dir - side - up, normal, color, glm::vec2{0, 0}));
            segment.addVertex(Mesh::VertexData(dir - side + up, normal, color, glm::vec2{0, 0}));
            segment.addVertex(Mesh::VertexData(dir + side - up, normal, color, glm::vec2{0, 0}));
            segment.addFace();
            segment.addVertex(Mesh::VertexData(dir + side + up, normal, color, glm::vec2{0, 0}));
            segment.addFaceRev();
        }
    }
    ret.addSegment(std::move(segment));
    return ret;
}

Mesh create_plane(const glm::vec3& origin, glm::vec3 normal, float size, const glm::vec3& color) {
    normal = glm::normalize(normal);
    glm::vec3 up =
      std::abs(normal.y) > std::abs(normal.z) ? glm::vec3{0, 0, 1} : glm::vec3{0, 1, 0};
    up -= normal * dot(up, normal);
    glm::vec3 side = glm::cross(normal, up);
    up *= size;
    side *= size;
    Mesh ret;
    Mesh::Segment segment;
    segment.addVertex(Mesh::VertexData(origin - side - up, normal, color, glm::vec2{0, 0}));
    segment.addVertex(Mesh::VertexData(origin - side + up, normal, color, glm::vec2{0, 1}));
    segment.addVertex(Mesh::VertexData(origin + side - up, normal, color, glm::vec2{1, 0}));
    segment.addFace();
    segment.addVertex(Mesh::VertexData(origin + side + up, normal, color, glm::vec2{1, 1}));
    segment.addFaceRev();
    ret.addSegment(std::move(segment));
    return ret;
}

Mesh create_sphere(size_t resX, size_t resY, float size, const glm::vec3& color) {
    Mesh ret;
    Mesh::Segment segment;
    for (size_t y = 0; y < resY; ++y) {
        float theta = static_cast<float>(y) / static_cast<float>(resY - 1);
        float fy = static_cast<float>(std::cos(static_cast<double>(theta) * M_PI));
        float fxz = static_cast<float>(std::sin(static_cast<double>(theta) * M_PI));
        for (size_t x = 0; x < resX; ++x) {
            float phi = static_cast<float>(x) / static_cast<float>(resX - 1);
            float fx = static_cast<float>(std::cos(static_cast<double>(phi * 2) * M_PI)) * fxz;
            float fz = static_cast<float>(std::sin(static_cast<double>(phi * 2) * M_PI)) * fxz;
            glm::vec3 normal{fx, fy, fz};
            glm::vec3 pos = normal * size;
            glm::vec2 texcoord{phi, 1.f - theta};
            segment.addVertex(Mesh::VertexData(pos, normal, color, texcoord));
        }
    }
    for (size_t y = 0; y < resY - 1; ++y) {
        for (size_t x = 0; x < resX - 1; ++x) {
            Mesh::VertexIndex v00 = static_cast<Mesh::VertexIndex>(x + y * resX);
            Mesh::VertexIndex v01 = static_cast<Mesh::VertexIndex>((x + 1) + y * resX);
            Mesh::VertexIndex v10 = static_cast<Mesh::VertexIndex>(x + (y + 1) * resX);
            Mesh::VertexIndex v11 = static_cast<Mesh::VertexIndex>((x + 1) + (y + 1) * resX);
            segment.addFace(v00, v01, v10);
            segment.addFace(v11, v01, v10);
        }
    }
    ret.addSegment(std::move(segment));
    return ret;
}
