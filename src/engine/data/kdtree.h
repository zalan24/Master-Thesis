#pragma once

#include <set>
#include <vector>

#include <glm/vec3.hpp>

class KDTree
{
 public:
    explicit KDTree(const std::vector<glm::vec3>& positions);
    explicit KDTree(std::vector<glm::vec3>&& positions);

    std::set<unsigned int> collectInSphere(const glm::vec3& position, double range) const;

 private:
    struct Point
    {
        unsigned int ind;
        unsigned int left;
        unsigned int right;
    };
    std::vector<glm::vec3> positions;
    std::vector<Point> indices;
    unsigned int root = 0;

    void build();
    unsigned int build(unsigned int a, unsigned int b, unsigned int depth);

    void collectInSphere(const glm::vec3& position, double range, unsigned int node,
                         std::set<unsigned int>& vec, unsigned int depth) const;

    static glm::vec3 getMask(unsigned int depth);
    static float maskPos(const glm::vec3& pos, const glm::vec3& mask);
    static float maskPos(const glm::vec3& pos, unsigned int depth);
};
