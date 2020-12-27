#include "sampler.h"

#include <thread>

#include "mappings.h"
#include "util.hpp"

Sampler::Sampler()
  : shaderProgram("sampler"),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER),
    modelTransform(1) {
    attributeBinder.addAttribute(&Sample::pos, "vPos");
    attributeBinder.addAttribute(&Sample::gridPos, "vGridPos");
    attributeBinder.addAttribute(&Sample::spherePos, "vSpherePos");
    attributeBinder.addAttribute(&Sample::polyPos, "vPolyPos");
    attributeBinder.addAttribute(&Sample::sideColor, "vSide");
    // attributeBinder.addAttribute(&Sample::tex, "vTex");
}

void Sampler::build(const BuildData& data, const SphereMesh* sphereMesh) {
    built = false;
    config = data;
    auto fix = [](uint& num) {
        if (num == 0)
            return;
        uint n = std::exchange(num, 1);
        while (num < n)
            num <<= 1;
    };
    fix(config.sphericalCoordinates);
    fix(config.cubeMap);
    fix(config.tetrahedron);
    fix(config.octahedron);

    std::vector<std::thread> threads;
    threads.push_back(std::thread([&, this] {
        sampleSC(data, sphereMesh, sphericalCoordinatesSamples, config.sphericalCoordinates);
    }));
    threads.push_back(
      std::thread([&, this] { sampleOH(data, sphereMesh, octahedronSamples, config.octahedron); }));

    for (std::thread& thr : threads)
        thr.join();
    built = true;
}

void Sampler::_render(const RenderContext& context) const {
    context.shaderManager->useProgram(shaderProgram);
    glBuffer.bind();
    glIndices.bind();
    attributeBinder.bind(*context.shaderManager);
    context.shaderManager->setUniform("PVM", context.pv * modelTransform);
    // context.shaderManager->setUniform("model", modelTransform);
    context.shaderManager->setUniform("lightColor", context.lightColor);
    context.shaderManager->setUniform("lightDir", context.lightDir);
    context.shaderManager->setUniform("ambientColor", context.ambientColor);
    context.shaderManager->setUniform("interpolation", interpolation);
    context.shaderManager->setUniform("numData", numData);
    context.shaderManager->setUniform("colorMode", displayCoords ? 1 : 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
}

void Sampler::sampleSC(const BuildData& data, const SphereMesh* sphereMesh,
                       std::vector<Sample>& vec, uint size) {
    vec.resize(size * size);
    for (uint x = 0; x < size; ++x) {
        float phi = static_cast<float>(x) / size * 2 * M_PI;
        for (uint y = 0; y < size; ++y) {
            float theta = static_cast<float>(y) / size * M_PI;
            SphereMesh::VertexData s = sphereMesh->getSample(glm::vec2{theta, phi});
            std::size_t ind = index(x, y, size);
            vec[ind].pos = s.position;
            vec[ind].gridPos =
              glm::vec2(static_cast<float>(y) / size, static_cast<float>(x) / size);
            vec[ind].spherePos = vec[ind].polyPos = sphericalToCartesion(
              glm::vec2{vec[ind].gridPos.x * M_PI, vec[ind].gridPos.y * 2 * M_PI});
            vec[ind].tex = s.texcoord;
            vec[ind].sideColor =
              glm::vec3{static_cast<float>(x) / size, static_cast<float>(y) / size, 0};
        }
    }
}

void Sampler::sampleOH(const BuildData& data, const SphereMesh* sphereMesh,
                       std::vector<Sample>& vec, uint uSize) {
    vec.resize(uSize * uSize);
    int size = static_cast<int>(uSize);
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            glm::vec3 uvw;
            Triangle tri;
            // int side = x * 4 / size;
            int side = 0;
            if (x <= size / 2) {
                if (y <= size / 2)
                    side = 0;
                else
                    side = 1;
            }
            else {
                if (y <= size / 2)
                    side = 3;
                else
                    side = 2;
            }
            glm::mat2 m;
            m[0][0] = 0;
            m[0][1] = -1;
            m[1][0] = 1;
            m[1][1] = 0;
            auto toVec3 = [](const glm::vec2& v) { return glm::vec3{v.x, 0, v.y}; };
            glm::vec2 p{0, -1};
            for (int i = 0; i < side; ++i)
                p = m * p;
            tri.b = toVec3(p);
            glm::vec3 btp = toVec3(p * (size / 2.0f) + size / 2.0f);
            p = m * p;
            tri.c = toVec3(p);
            glm::vec3 ctp = toVec3(p * (size / 2.0f) + size / 2.0f);
            int steps = std::abs(x - size / 2) + std::abs(y - size / 2);
            glm::vec3 atp;
            if (steps <= size / 2) {  // top
                tri.a = glm::vec3{0, 1, 0};
                atp = glm::vec3{size / 2, 0, size / 2};
            }
            else {  // bottom
                tri.a = glm::vec3{0, -1, 0};
                atp = glm::vec3{x <= size / 2 ? 0 : size, 0, y <= size / 2 ? 0 : size};
            }
            glm::vec3 tp{x, 0, y};
            uvw.x = glm::length(glm::cross(tp - btp, ctp - btp));
            uvw.y = glm::length(glm::cross(tp - atp, ctp - atp));
            float a = glm::length(glm::cross(btp - atp, ctp - atp));
            uvw.x /= a;
            uvw.y /= a;
            uvw.z = 1 - uvw.x - uvw.y;
            glm::vec2 sc = triangleToSphere(tri, uvw);
            SphereMesh::VertexData s = sphereMesh->getSample(sc);
            if (sc.y < 0)
                sc.y += M_PI * 2;
            std::size_t ind = index(x, y, size);
            vec[ind].pos = s.position;
            vec[ind].gridPos =
              glm::vec2(static_cast<float>(y) / size, static_cast<float>(x) / size);
            vec[ind].spherePos = sphericalToCartesion(sc);
            vec[ind].polyPos = tri.a * uvw.x + tri.b * uvw.y + tri.c * uvw.z;
            vec[ind].tex = s.texcoord;
            vec[ind].sideColor =
              glm::vec3{tri.b.x / 2.0f + 0.5f, tri.b.z / 2.0f + 0.5f, tri.a.y / 2.0f + 0.5f};
        }
    }
}

void Sampler::display(MapMode mode) {
    indices.clear();
    displayMode = mode;
    auto addTriangle = [this](VertexIndex p1, VertexIndex p2, VertexIndex p3) {
        indices.push_back(p1);
        indices.push_back(p2);
        indices.push_back(p3);
    };
    switch (mode) {
        case MapMode::SPHERICAL_COORDINATES:
            glBuffer.upload(sphericalCoordinatesSamples, GL_STATIC_DRAW);
            for (uint x = 0; x < config.sphericalCoordinates - 1; ++x) {
                for (uint y = 0; y < config.sphericalCoordinates - 1; ++y) {
                    addTriangle(
                      index(x, y, config.sphericalCoordinates),
                      index((x + 1) % config.sphericalCoordinates, y, config.sphericalCoordinates),
                      index(x, (y + 1) % config.sphericalCoordinates, config.sphericalCoordinates));
                    addTriangle(
                      index((x + 1) % config.sphericalCoordinates, y, config.sphericalCoordinates),
                      index(x, (y + 1) % config.sphericalCoordinates, config.sphericalCoordinates),
                      index((x + 1) % config.sphericalCoordinates,
                            (y + 1) % config.sphericalCoordinates, config.sphericalCoordinates));
                }
            }
            numData = config.sphericalCoordinates;
            break;
        case MapMode::CUBE_MAP:
            glBuffer.upload(cubeMapSamples, GL_STATIC_DRAW);
            numData = config.cubeMap;
            break;
        case MapMode::TETRAHEDRON:
            glBuffer.upload(tetrahedronSamples, GL_STATIC_DRAW);
            numData = config.tetrahedron;
            break;
        case MapMode::OCTAHEDRON:
            glBuffer.upload(octahedronSamples, GL_STATIC_DRAW);
            for (uint x = 0; x < config.octahedron - 1; ++x) {
                for (uint y = 0; y < config.octahedron - 1; ++y) {
                    addTriangle(index(x, y, config.octahedron),
                                index((x + 1) % config.octahedron, y, config.octahedron),
                                index(x, (y + 1) % config.octahedron, config.octahedron));
                    addTriangle(index((x + 1) % config.octahedron, y, config.octahedron),
                                index(x, (y + 1) % config.octahedron, config.octahedron),
                                index((x + 1) % config.octahedron, (y + 1) % config.octahedron,
                                      config.octahedron));
                }
            }
            numData = config.octahedron;
            break;
    }
    glIndices.upload(indices, GL_STATIC_DRAW);
}

std::vector<FloatOption> Sampler::getOptions() {
    return {FloatOption{"interpolation", &interpolation}};
}

std::vector<BoolOption> Sampler::getBoolOptions() {
    return {BoolOption{"display coordinates", &displayCoords}};
}

bool Sampler::isBuilt() const {
    return built;
}

void Sampler::write(std::ostream& out) const {
    writeData(out, config);
    writeData(out, modelTransform);

    writeData(out, sphericalCoordinatesSamples);
    writeData(out, cubeMapSamples);
    writeData(out, tetrahedronSamples);
    writeData(out, octahedronSamples);

    // writeData(out, indices);

    writeData(out, numData);
    writeData(out, interpolation);
    writeData(out, built);
    writeData(out, displayMode);
    writeData(out, displayCoords);
}

void Sampler::read(std::istream& in) {
    // readData(in, config);
    // readData(in, modelTransform);

    // readData(in, sphericalCoordinatesSamples);
    // readData(in, cubeMapSamples);
    // readData(in, tetrahedronSamples);
    // readData(in, octahedronSamples);

    // // readData(in, indices);

    // readData(in, numData);
    // readData(in, interpolation);
    // readData(in, built);
    // readData(in, displayMode);
    // readData(in, displayCoords);
    // if (displayMode != MapMode::NONE) {
    //     display(displayMode);
    // }
}
