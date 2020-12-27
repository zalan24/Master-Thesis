#include "spheremesh.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "intersections.h"
#include "mappings.h"
#include "kdtree.h"
#include "util.hpp"

float SphereMesh::sMatrixAlpha_option = 0;
float SphereMesh::sShoothenBias_option = 0.25;
int SphereMesh::sPreprocessCount_option = 200;
int SphereMesh::sProcessMax_option = 1000;

SphereMesh::SphereMesh(const Mesh& mesh, const std::string& s)
  : shaderProgram(s),
    outlinerProgram("outline"),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER),
    modelTransform(1) {
    bindAttributes();

    compile(mesh);
}

SphereMesh::SphereMesh(const std::string& s)
  : shaderProgram(s),
    outlinerProgram("outline"),
    glBuffer(GL_ARRAY_BUFFER),
    glIndices(GL_ELEMENT_ARRAY_BUFFER),
    modelTransform(1) {
    bindAttributes();
}

SphereMesh::~SphereMesh() {
    terminateBuild();
    if (thread != nullptr) {
        if (thread->joinable()) {
            thread->join();
        }
    }
}

void SphereMesh::bindAttributes() {
    attributeBinder.addAttribute(&VertexData::position, "vPos");
    attributeBinder.addAttribute(&VertexData::normal, "vNorm", true);
    attributeBinder.addAttribute(&VertexData::color, "vCol");
    attributeBinder.addAttribute(&VertexData::new_position, "vNewPos");

    outlineAttributeBinder.addAttribute(&VertexData::new_position, "vNewPos");
    outlineAttributeBinder.addAttribute(&VertexData::position, "vPos");
}

void SphereMesh::_render(const RenderContext& context) const {
    if (status == Status::VERTEX_DATA_READY) {
        // ...
        uploadData();
        status = Status::UPLOADED;
        thread.reset(nullptr);
    }
    else if (status != Status::UPLOADED)
        return;
    context.shaderManager->useProgram(shaderProgram);
    glBuffer.bind();
    glIndices.bind();
    attributeBinder.bind(*context.shaderManager);
    context.shaderManager->setUniform("PVM", context.pv * modelTransform);
    context.shaderManager->setUniform("model", modelTransform);
    context.shaderManager->setUniform("interpolation", interpolation);
    context.shaderManager->setUniform("lightColor", context.lightColor);
    context.shaderManager->setUniform("lightDir", context.lightDir);
    context.shaderManager->setUniform("ambientColor", context.ambientColor);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, index.size(), GL_UNSIGNED_INT, 0);

    context.shaderManager->useProgram(outlinerProgram);
    context.shaderManager->setUniform("PVM", context.pv * modelTransform);
    context.shaderManager->setUniform("color", glm::vec3{1, 0, 0});
    context.shaderManager->setUniform("interpolation", interpolation);
    glBuffer.bind();
    glIndices.bind();
    outlineAttributeBinder.bind(*context.shaderManager);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_TRIANGLES, index.size(), GL_UNSIGNED_INT, 0);
}

void SphereMesh::addTriangle(Mesh::VertexIndex ind, Mesh::VertexIndex n1, Mesh::VertexIndex n2,
                             const glm::vec3& indPos, const glm::vec3& n1Pos,
                             const glm::vec3& n2Pos) {
    if (ind == n1 || ind == n2 || n1 == n2)
        return;
    FaceIndex f = index.size() / 3;
    nodes[ind]->e[n1] = {glm::distance(indPos, n1Pos)};
    nodes[ind]->e[n2] = {glm::distance(indPos, n2Pos)};
    nodes[ind]->f.insert(f);

    nodes[n1]->e[n2] = {glm::distance(n1Pos, n2Pos)};
    nodes[n1]->e[ind] = {glm::distance(n1Pos, indPos)};
    nodes[n1]->f.insert(f);

    nodes[n2]->e[ind] = {glm::distance(n2Pos, indPos)};
    nodes[n2]->e[n1] = {glm::distance(n2Pos, n1Pos)};
    nodes[n2]->f.insert(f);

    index.push_back(ind);
    index.push_back(n1);
    index.push_back(n2);
}

void SphereMesh::buildGraph(const Mesh& mesh) {
    nodes.clear();
    nodes.resize(mesh.vertices.size());
    remap.resize(mesh.vertices.size());
    std::vector<glm::vec3> positions(mesh.vertices.size());
    for (int i = 0; i < positions.size(); ++i) {
        positions[i] = mesh.vertices[i].position;
        remap[i] = i;
        nodes[i].reset(new NodeData{});
        nodes[i]->originalPosition = mesh.vertices[i].position;
        nodes[i]->position = mesh.vertices[i].position;
        nodes[i]->color = mesh.vertices[i].color;
        nodes[i]->texcoord = mesh.vertices[i].texcoord;
    }
    KDTree tree{std::move(positions)};
    float closeRange = 0.0001f;
    for (VertexIndex i = 0; i < mesh.vertices.size(); ++i) {
        if (remap[i] != i)
            continue;
        std::set<VertexIndex> v = tree.collectInSphere(mesh.vertices[i].position, closeRange);
        for (const VertexIndex& ind : v) {
            if (ind == i)
                continue;
            assert(glm::distance(mesh.vertices[ind].position, mesh.vertices[i].position)
                   < closeRange);
            remap[ind] = i;
        }
    }
    auto add = [&, this](Mesh::VertexIndex ind, Mesh::VertexIndex n1, Mesh::VertexIndex n2) {
        // assert(ind != n1 && ind != n2);
        addTriangle(ind, n1, n2, mesh.vertices[ind].position, mesh.vertices[n1].position,
                    mesh.vertices[n2].position);
    };
    index.clear();
    for (std::size_t ind = 0; ind < mesh.index.size(); ind += 3) {
        add(remap[mesh.index[ind]], remap[mesh.index[ind + 1]], remap[mesh.index[ind + 2]]);
    }
    // index.resize(mesh.index.size());
    // for (std::size_t ind = 0; ind < mesh.index.size(); ++ind) {
    //     index[ind] = remap[mesh.index[ind]];
    // }
}

std::vector<std::tuple<SphereMesh::VertexIndex, SphereMesh::VertexIndex>> SphereMesh::getHoleEdges(
  VertexIndex i, VertexIndex j) const {
    std::vector<std::tuple<VertexIndex, VertexIndex>> ret;
    VertexIndex a = i;
    VertexIndex b = j;
    bool found = false;
    do {
        found = false;
        ret.push_back(std::make_tuple(a, b));
        for (const auto& [ind, data] : nodes[b]->e) {
            if (ind == a)
                continue;
            if (combination(nodes[b]->f, nodes[ind]->f, CombinationBits::INCLUDE_BOTH_BIT).size()
                == 1) {
                a = std::exchange(b, ind);
                found = true;
                break;
            }
        }
    } while (b != j && found);
    return ret;
}

void SphereMesh::repair(const Mesh& mesh) {
    std::cout << "repairing mesh..." << std::endl;
    for (VertexIndex i = 0, count = nodes.size(); i < count; ++i) {
        for (const auto& [j, dat] : nodes[i]->e) {
            std::set<VertexIndex> section =
              combination(nodes[i]->f, nodes[j]->f, CombinationBits::INCLUDE_BOTH_BIT);
            if (section.size() == 1) {
                // side of the mesh, assumed to be a hole
                std::vector<std::tuple<VertexIndex, VertexIndex>> edges = getHoleEdges(i, j);
                assert(edges.size() > 0);
                if (std::get<1>(edges.back()) != std::get<0>(edges[0]))
                    continue;
                glm::vec3 G{0, 0, 0};
                glm::vec2 TG{0, 0};
                VertexIndex newNode = nodes.size();
                nodes.push_back(std::make_unique<NodeData>());
                remap.push_back(remap.size());
                for (const auto& [vi, vj] : edges) {
                    G += nodes[vi]->position;
                    TG += nodes[vi]->texcoord;
                }
                G /= edges.size();
                TG /= edges.size();
                nodes[newNode]->originalPosition = G;
                nodes[newNode]->position = G;
                nodes[newNode]->color = glm::vec3{0, 1, 0};
                nodes[newNode]->texcoord = TG;
                for (const auto& [vi, vj] : edges) {
                    addTriangle(newNode, vi, vj, G, nodes[vi]->position, nodes[vj]->position);
                }
                std::cout << "hole fixed: " << i << std::endl;
            }
        }
    }
    std::cout << "done." << std::endl;
}

void SphereMesh::createVertexData() {
    vertices.clear();
    vertices.resize(nodes.size());
    for (std::size_t ind = 0; ind < nodes.size(); ++ind) {
        vertices[ind] = VertexData{nodes[ind]->originalPosition, glm::vec3{0, 0, 0},
                                   nodes[ind]->color, nodes[ind]->texcoord, nodes[ind]->position};
    }
    isOverlapping(true);
    bvTree.build(index.size() / 3, 10, [this](unsigned int i) {
        Triangle tri = getTriangle(i);
        return ::getBoundingSphere(tri);
    });
}

void SphereMesh::uploadData() const {
    glBuffer.upload(vertices, GL_STATIC_DRAW);
    glIndices.upload(index, GL_STATIC_DRAW);
}

void SphereMesh::compile(const Mesh& mesh) {
    if (!mesh.built)
        throw std::runtime_error("SphereMesh can only be constructed from a built mesh");
    status = Status::UNINITIALIZED;
    buildGraph(mesh);
    status = Status::GRAPH_READY;
    repair(mesh);
    status = Status::REPAIRED;
    refine();
}

void SphereMesh::rebuild() {
    if (status != Status::UPLOADED)
        throw std::runtime_error("Cannot rebuild the spheremesh: its status has to be 'UPLOADED'");
    status = Status::REPAIRED;
    refine();
}

void SphereMesh::refine() {
    if (status == Status::UNINITIALIZED)
        throw std::runtime_error("Cannot refine the spheremesh: it is uninitialized");
    matrixAlpha = sMatrixAlpha_option;
    shoothenBias = sShoothenBias_option;
    preprocessCount = sPreprocessCount_option;
    processMax = sProcessMax_option;
    thread = std::make_unique<std::thread>(std::thread([this] {
        if (status == Status::REPAIRED) {
            createMatrix(matrixAlpha);
            status = Status::MATRIX_READY;
        }
        if (status == Status::MATRIX_READY)
            preprocess();
        status = Status::PREPROCESSED;  // set it ether way

        // process(mesh, 100);
        smartProcess();
        status = Status::PROCESSED;
        createVertexData();
        status = Status::VERTEX_DATA_READY;
    }));
    thread->detach();
}

void SphereMesh::preprocess() {
    smoothen(0, preprocessCount);
    glm::vec3 G{0, 0, 0};
    int count = 0;
    for (VertexIndex i = 0; i < nodes.size(); ++i) {
        if (remap[i] != i)
            continue;
        G += nodes[i]->position;
        count++;
    }
    G /= count;
    glm::vec3 C{0, 0, 0};
    double divider = 0;
    for (VertexIndex i = 0; i < nodes.size(); ++i) {
        if (remap[i] != i)
            continue;
        float len = glm::length(nodes[i]->position - G);
        divider += len;
        C += len * nodes[i]->position;
    }
    C /= divider;
    project(C);
}

void SphereMesh::process(int repeat) {
    for (int i = 0; i < repeat; ++i) {
        smoothen(shoothenBias);
        adjust();
        project(glm::vec3{0, 0, 0});
    }
}

void SphereMesh::adjust() {
    temp_positions.resize(nodes.size());
    for (VertexIndex i = 0; i < nodes.size(); ++i) {
        if (remap[i] != i)
            continue;
        float A = 0;
        glm::vec3 F{0, 0, 0};
        for (const FaceIndex& f : nodes[i]->f) {
            const glm::vec3 p1 = nodes[remap[index[f * 3]]]->position;
            const glm::vec3 p2 = nodes[remap[index[f * 3 + 1]]]->position;
            const glm::vec3 p3 = nodes[remap[index[f * 3 + 2]]]->position;
            const glm::vec3 Fi = (p1 + p2 + p3) / 3.0f;
            const glm::vec3 a = p2 - p1;
            const glm::vec3 b = p3 - p1;
            float Ai = glm::length(glm::cross(a, b)) / 2;  // |axb| = |a|*|b|*sin(AB) = 2*area
            F += Fi * Ai;
            A += Ai;
        }
        if (A > 0)
            temp_positions[i] = F / A;
        else
            temp_positions[i] = glm::vec3{0, 0, 0};
    }
    for (VertexIndex i = 0; i < nodes.size(); ++i) {
        nodes[i]->position = temp_positions[i];
    }
}

void SphereMesh::project(const glm::vec3& from, float size) {
    for (std::unique_ptr<NodeData>& node : nodes) {
        glm::vec3 d = node->position - from;
        if (glm::dot(d, d) > 0.00001)
            node->position = glm::normalize(d) * size;
    }
}

std::vector<std::string> SphereMesh::getPrograms() const {
    return {shaderProgram, outlinerProgram};
}

std::vector<FloatOption> SphereMesh::getOptions() {
    return {FloatOption{"interpolation", &interpolation}};
}

void SphereMesh::smoothen(float extra, int repeat) {
    for (int i = 0; i < repeat; ++i) {
        smoothen(extra);
    }
}

void SphereMesh::smoothen(float extra) {
    temp_positions.resize(nodes.size());
    const std::size_t group = (nodes.size() + MAX_THREADS - 1) / MAX_THREADS;
    std::vector<std::thread> threads(MAX_THREADS);
    for (int thr = 0; thr < MAX_THREADS; ++thr) {
        threads[thr] = std::thread([&group, thr, this] {
            for (std::size_t i = group * thr; i < nodes.size() && i < group * (thr + 1); ++i) {
                temp_positions[i] = glm::vec3{0, 0, 0};
                for (const auto& [j, value] : A[i]) {
                    temp_positions[i] += value * nodes[j]->position;
                }
            }
        });
    }
    for (std::thread& thr : threads)
        thr.join();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        nodes[i]->position = temp_positions[i] + (temp_positions[i] - nodes[i]->position) * extra;
    }
}

void SphereMesh::createMatrix(float alpha) {
    A.clear();
    A.resize(nodes.size());
    for (std::size_t ind = 0; ind < nodes.size(); ++ind) {
        float ni = nodes[ind]->e.size();
        auto add = [&, this](VertexIndex j, float bij) {
            if (ni > 0)
                A[ind][j] = bij / (ni * (ni + 5));
        };
        float dij = 3;  // every face has 3 sides
        float dji = 3;  // every face has 3 sides
        float bii = (ni - 1) * ni + ni * alpha + 4.0f / dij * ni;
        add(ind, bii);
        for (const auto& [j, edgeData] : nodes[ind]->e) {
            assert(j != ind);
            float bij = (2 - alpha + 4.0f / dij + 4.0f / dji);
            add(j, bij);
        }
    }
}

void SphereMesh::smartProcess() {
    progress = 0;
    int count = 0;
    float steps = 10;
    float incRate = 1.0f / 5;
    do {
        int rep = static_cast<int>(steps);
        std::cout << "Smart process: doing " << rep << " steps, " << count << " done" << std::endl;
        process(rep);
        count += rep;
        steps += incRate * steps;
        progress = count;
    } while (count < processMax && isOverlapping());
    process(count / 10);
}

bool SphereMesh::isOverlapping(bool paint) {
    bool ret = false;
    temp_positions.resize(nodes.size());
    for (VertexIndex i = 0; i < nodes.size(); ++i) {
        temp_positions[i] = nodes[i]->position;
    }
    KDTree tree{temp_positions};
    for (std::size_t ind = 0; ind < index.size(); ind += 3) {
        VertexIndex v1 = remap[index[ind]];
        glm::vec3 p1 = nodes[v1]->position;
        VertexIndex v2 = remap[index[ind + 1]];
        glm::vec3 p2 = nodes[v2]->position;
        VertexIndex v3 = remap[index[ind + 2]];
        glm::vec3 p3 = nodes[v3]->position;
        // if (glm::length(glm::cross(p2 - p1, p3 - p1)) / 2 <= 0.01 / mesh.vertices.size())
        //     continue;
        std::set<VertexIndex> n;
        glm::vec3 G = (p1 + p2 + p3) / 3.f;
        float r = std::max(glm::length(G - p1), std::max(glm::length(G - p2), glm::length(G - p3)));
        n = tree.collectInSphere(G, r);
        n.erase(v1);
        n.erase(v2);
        n.erase(v3);
        for (const VertexIndex& j : n) {
            if (remap[j] != j)
                continue;
            glm::vec3 jPos = nodes[j]->position;
            glm::vec3 N = glm::cross(p2 - p1, p3 - p1);
            float t = glm::dot(N, p1) / glm::dot(N, jPos);
            glm::vec3 pos = t * jPos;  // projected to triangle
            float val1 = glm::dot(glm::cross(pos - p1, p2 - p1), N);
            float val2 = glm::dot(glm::cross(pos - p2, p3 - p2), N);
            float val3 = glm::dot(glm::cross(pos - p3, p1 - p3), N);
            if (val1 * val2 < 0 || val1 * val3 < 0)
                continue;
            if (paint) {
                vertices[v1].color = glm::vec3{1, 0, 0};
                vertices[v2].color = glm::vec3{1, 0, 0};
                vertices[v3].color = glm::vec3{1, 0, 0};
                ret = true;
            }
            else
                return true;
        }
    }
    return ret;
}

void SphereMesh::terminateBuild() {
    preprocessCount = 0;
    processMax = 0;
}

void SphereMesh::write(std::ostream& out) const {
    writeData(out, status);
    writeData(out, modelTransform);

    writeData(out, nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        writeData(out, nodes[i]->e);
        writeData(out, nodes[i]->f);
        writeData(out, nodes[i]->position);
        writeData(out, nodes[i]->originalPosition);
        writeData(out, nodes[i]->color);
        writeData(out, nodes[i]->texcoord);
    }

    writeData(out, remap);
    writeData(out, A);
    writeData(out, vertices);
    writeData(out, index);
}

void SphereMesh::read(std::istream& in) {
    readData(in, status);
    readData(in, modelTransform);

    decltype(nodes.size()) len;
    readData(in, len);
    nodes.resize(len);
    for (std::size_t i = 0; i < len; ++i) {
        nodes[i].reset(new NodeData{});
        readData(in, nodes[i]->e);
        readData(in, nodes[i]->f);
        readData(in, nodes[i]->position);
        readData(in, nodes[i]->originalPosition);
        readData(in, nodes[i]->color);
        readData(in, nodes[i]->texcoord);
    }

    readData(in, remap);
    readData(in, A);
    readData(in, vertices);
    readData(in, index);

    if (status == Status::VERTEX_DATA_READY || status == Status::UPLOADED)
        createVertexData();
    if (status == Status::UPLOADED)
        uploadData();
}

Triangle SphereMesh::getTriangle(std::size_t ind) const {
    ind *= 3;
    VertexIndex v1 = remap[index[ind]];
    glm::vec3 p1 = nodes[v1]->position;
    VertexIndex v2 = remap[index[ind + 1]];
    glm::vec3 p2 = nodes[v2]->position;
    VertexIndex v3 = remap[index[ind + 2]];
    glm::vec3 p3 = nodes[v3]->position;
    return Triangle{p1, p2, p3};
}

SphereMesh::VertexData SphereMesh::getSample(const glm::vec2& sc) const {
    if (!bvTree.isReady())
        throw std::runtime_error("bvTree has not yet been built");
    glm::vec4 bestMatch{0, 0, 0, 0};
    float bestVal = -1;  // pick the biggest
    std::size_t bestInd = 0;
    const Ray ray{glm::vec3{0, 0, 0},
                  sphericalToCartesion(sc)};
    bvTree.intersect(ray, [&, this](std::size_t ind) {
        Triangle tri = getTriangle(ind);
        glm::vec4 r = intersect(ray, tri);
        if (r.w > 0) {
            glm::vec3 c = glm::cross(tri.b - tri.a, tri.c - tri.a);
            float val = glm::dot(c, c);
            if (val > bestVal) {
                bestVal = val;
                bestMatch = r;
                bestInd = ind;
            }
        }
    });
    VertexData ret;
    if (bestVal > 0) {
        bestInd *= 3;
        VertexIndex v1 = remap[index[bestInd]];
        VertexIndex v2 = remap[index[bestInd + 1]];
        VertexIndex v3 = remap[index[bestInd + 2]];
        ret.position = vertices[v1].position * bestMatch.x + vertices[v2].position * bestMatch.y
                       + vertices[v3].position * bestMatch.z;
        ret.normal = vertices[v1].normal * bestMatch.x + vertices[v2].normal * bestMatch.y
                     + vertices[v3].normal * bestMatch.z;
        ret.color = vertices[v1].color * bestMatch.x + vertices[v2].color * bestMatch.y
                    + vertices[v3].color * bestMatch.z;
        ret.texcoord = vertices[v1].texcoord * bestMatch.x + vertices[v2].texcoord * bestMatch.y
                       + vertices[v3].texcoord * bestMatch.z;
        ret.new_position = vertices[v1].new_position * bestMatch.x
                           + vertices[v2].new_position * bestMatch.y
                           + vertices[v3].new_position * bestMatch.z;
    }
    else {
        // throw std::runtime_error("Could not find intersection with ray");
        ret.position = glm::vec3{0, 0, 0};
        ret.normal = glm::vec3{0, 0, 0};
        ret.color = glm::vec3{1, 0, 0};
        ret.texcoord = glm::vec2{1, 0};
        ret.new_position = glm::vec3{0, 0, 0};
    }
    return ret;
}
