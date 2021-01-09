#include "kdtree.h"

#include <algorithm>
#include <iterator>
#include <utility>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

KDTree::KDTree(const std::vector<glm::vec3>& poss) : positions(poss) {
    build();
}

KDTree::KDTree(std::vector<glm::vec3>&& poss) : positions(std::move(poss)) {
    build();
}

void KDTree::build() {
    indices.clear();
    indices.resize(positions.size());
    for (unsigned int i = 0; i < indices.size(); ++i) {
        indices[i].ind = i;
        indices[i].left = static_cast<unsigned int>(-1);
        indices[i].right = static_cast<unsigned int>(-1);
    }
    root = build(0, static_cast<unsigned int>(positions.size()), 0);
}

unsigned int KDTree::build(unsigned int a, unsigned int b, unsigned int depth) {
    if (a == b)
        return static_cast<unsigned int>(-1);
    if (a + 1 == b)
        return a;
    glm::vec3 mask = getMask(depth);
    std::sort(std::next(std::begin(indices), a), std::next(std::begin(indices), b),
              [&mask, this](const Point& lhs, const Point& rhs) {
                  return maskPos(positions[lhs.ind], mask) < maskPos(positions[rhs.ind], mask);
              });
    unsigned int mid = (a + b) / 2;
    indices[mid].left = build(a, mid, depth + 1);
    indices[mid].right = build(mid + 1, b, depth + 1);
    return mid;
}

std::set<unsigned int> KDTree::collectInSphere(const glm::vec3& position, double range) const {
    std::set<unsigned int> ret;
    collectInSphere(position, range, root, ret, 0);
    return ret;
}

void KDTree::collectInSphere(const glm::vec3& position, double range, unsigned int node,
                             std::set<unsigned int>& vec, unsigned int depth) const {
    if (node == static_cast<unsigned int>(-1))
        return;
    glm::vec3 d = position - positions[indices[node].ind];
    if (glm::dot(d, d) < static_cast<float>(range * range))
        vec.insert(indices[node].ind);
    float value = maskPos(d, depth);
    if (value < static_cast<float>(range))
        collectInSphere(position, range, indices[node].left, vec, depth + 1);
    if (value > static_cast<float>(-range))
        collectInSphere(position, range, indices[node].right, vec, depth + 1);
}

glm::vec3 KDTree::getMask(unsigned int depth) {
    glm::vec3 mask{0, 0, 0};
    mask[depth % 3] = 1;
    return mask;
}

float KDTree::maskPos(const glm::vec3& pos, const glm::vec3& mask) {
    return glm::dot(pos, mask);
}

float KDTree::maskPos(const glm::vec3& pos, unsigned int depth) {
    return maskPos(pos, getMask(depth));
}
