#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

#include <intersections.h>

template <typename I>
class SphereTree
{
 private:
    class SphereTreeNode
    {
     public:
        std::unique_ptr<SphereTreeNode> left;
        std::unique_ptr<SphereTreeNode> right;
        std::size_t dataStart = 0;
        std::size_t dataCount = 0;
        Sphere bound;

        static Sphere getBound(const Sphere& s1, const Sphere& s2) {
            Sphere ret;
            glm::vec3 pos1 = s1.getPos();
            glm::vec3 pos2 = s2.getPos();
            glm::vec3 d = pos2 - pos1;
            float len2 = glm::dot(d, d);
            if (len2 <= (s1.pos_radius.w - s2.pos_radius.w) * (s1.pos_radius.w - s2.pos_radius.w)) {
                Sphere ret = s1.pos_radius.w > s2.pos_radius.w ? s1 : s2;
                return ret;
            }
            d /= sqrtf(len2);
            glm::vec3 p1 = pos1 - d * s1.pos_radius.w;
            glm::vec3 p2 = pos2 + d * s2.pos_radius.w;
            glm::vec3 o = (p1 + p2) / 2.0f;
            ret.pos_radius = glm::vec4{o.x, o.y, o.z, glm::distance(p1, p2) / 2};
            return ret;
        }

        template <typename F>  // F : I -> BoundingSphere
        static std::unique_ptr<SphereTreeNode> build(std::size_t count, std::vector<I>& indices,
                                                     std::vector<Sphere>& bounds,
                                                     std::size_t maxCountInNode, F&& f) {
            indices.resize(count);
            bounds.resize(count);
            for (I i = 0; i < count; ++i) {
                indices[i] = i;
                bounds[i] = f(i);
            }
            return build(0, count, indices, bounds, maxCountInNode);
        }

        template <typename T, typename F>  // F : T -> BoundingSphere
        static std::unique_ptr<SphereTreeNode> build(const std::vector<T>& vec,
                                                     std::vector<std::size_t>& indices,
                                                     std::vector<Sphere>& bounds,
                                                     std::size_t maxCountInNode, F&& f) {
            return build(vec.size(), indices, bounds, maxCountInNode,
                         [&](std::size_t i) { return f(vec[i]); });
        }

        template <typename F>
        void intersect(const Ray& r, const std::vector<I>& indices, F&& f) const {
            if (::intersect(r, bound, false) >= 0) {
                for (I i = static_cast<I>(dataStart); i < dataStart + dataCount; ++i) {
                    f(indices[i]);
                }
                if (left)
                    left->intersect(r, indices, f);  // no forwarding
                if (right)
                    right->intersect(r, indices, f);  // no forwarding
            }
        }

     private:
        static std::unique_ptr<SphereTreeNode> build(std::size_t begin, std::size_t end,
                                                     std::vector<I>& indices,
                                                     std::vector<Sphere>& bounds,
                                                     std::size_t maxCountInNode) {
            std::size_t count = end - begin;
            if (count <= 0)
                return nullptr;
            if (count <= maxCountInNode || count == 1) {
                Sphere bound = bounds[indices[begin]];
                for (I i = begin + 1; i < end; ++i) {
                    bound = getBound(bound, bounds[indices[i]]);
                }
                std::unique_ptr<SphereTreeNode> ret = std::make_unique<SphereTreeNode>();
                ret->dataStart = begin;
                ret->dataCount = count;
                ret->bound = bound;
                return ret;
            }
            std::unique_ptr<SphereTreeNode> ret = std::make_unique<SphereTreeNode>();
            ret->dataStart = 0;
            ret->dataCount = 0;
            Sphere bound = bounds[indices[begin]];
            glm::vec3 mn{bound.pos_radius.x + bound.pos_radius.w,
                         bound.pos_radius.y + bound.pos_radius.w,
                         bound.pos_radius.z + bound.pos_radius.w};
            glm::vec3 mx{bound.pos_radius.x - bound.pos_radius.w,
                         bound.pos_radius.y - bound.pos_radius.w,
                         bound.pos_radius.z - bound.pos_radius.w};
            for (I i = begin + 1; i < end; ++i) {
                bound = bounds[indices[i]];
                glm::vec3 n{bound.pos_radius.x + bound.pos_radius.w,
                            bound.pos_radius.y + bound.pos_radius.w,
                            bound.pos_radius.z + bound.pos_radius.w};
                glm::vec3 x{bound.pos_radius.x - bound.pos_radius.w,
                            bound.pos_radius.y - bound.pos_radius.w,
                            bound.pos_radius.z - bound.pos_radius.w};
                mn = glm::min(mn, n);
                mx = glm::max(mx, x);
            }
            glm::vec3 d = mx - mn;
            auto vBegin = std::next(std::begin(indices), begin);
            auto vEnd = std::next(std::begin(indices), end);
            if (d.x > d.y && d.x > d.z) {
                std::sort(vBegin, vEnd, [&](const I& lhs, const I& rhs) {
                    return bounds[lhs].pos_radius.x < bounds[rhs].pos_radius.x;
                });
            }
            else if (d.y > d.z) {
                std::sort(vBegin, vEnd, [&](const I& lhs, const I& rhs) {
                    return bounds[lhs].pos_radius.y < bounds[rhs].pos_radius.y;
                });
            }
            else {
                std::sort(vBegin, vEnd, [&](const I& lhs, const I& rhs) {
                    return bounds[lhs].pos_radius.z < bounds[rhs].pos_radius.z;
                });
            }
            I m = begin + count / 2;
            ret->left = build(begin, m, indices, bounds, maxCountInNode);
            ret->right = build(m, end, indices, bounds, maxCountInNode);
            if (ret->left) {
                ret->bound = ret->left->bound;
                if (ret->right)
                    ret->bound = getBound(ret->bound, ret->right->bound);
            }
            else if (ret->right)
                ret->bound = ret->right->bound;
            if (ret->left)
                assert(isInside(ret->left->bound, ret->bound));
            if (ret->right)
                assert(isInside(ret->right->bound, ret->bound));
            return std::move(ret);
        }
    };

 public:
    bool isReady() const { return root != nullptr; }

    template <typename F>  // F : I -> BoundingSphere
    void build(std::size_t count, std::size_t maxCountInNode, F&& f) {
        root = SphereTreeNode::build(count, indices, bounds, maxCountInNode, std::forward<F>(f));
    }

    template <typename T, typename F>  // F : T -> BoundingSphere
    void build(const std::vector<T>& vec, std::size_t maxCountInNode, F&& f) {
        root =
          SphereTreeNode::build(count, vec, indices, bounds, maxCountInNode, std::forward<F>(f));
    }

    template <typename F>
    void intersect(const Ray& r, F&& f) const {
        root->intersect(r, indices, std::forward<F>(f));
    }

 private:
    std::vector<I> indices;
    std::vector<Sphere> bounds;
    std::unique_ptr<SphereTreeNode> root;
};
