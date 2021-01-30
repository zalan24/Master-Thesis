#pragma once

#include "matrix.hpp"

template <typename H>
class Camera
{
    static_assert(H::rows() == 3 && H::columns() == 4, "The camera matrix has to be 3x4");

 public:
    using Mat = H;
    using InvMat = decltype(std::declval<H>().transpose());
    Camera() = default;
    explicit Camera(const H& h) : data(h), inv(::pseudoInverse(h)) {}

    operator H() const { return data; }

    const H& matrix() const { return data; }
    const InvMat& inverseMatrix() const { return inv; }
    Camera inverse() const { return Camera(inv.transpose(), data.transpose()); }

    bool operator==(const Camera& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Camera& other) const { return !(*this == other); }

 private:
    H data;
    InvMat inv;

    explicit Camera(const H& h, const InvMat& i) : data(h), inv(i) {}
};

template <typename H>
Camera<H> centralProjection(const typename H::ValueType& focalLength) {
    H mat{0};
    mat.at(0u, 0u) = focalLength;
    mat.at(1u, 1u) = focalLength;
    mat.at(2u, 2u) = 1;
    return Camera<H>(mat);
}

template <typename H>
Camera<H> offsetProjection(const typename H::ValueType& focalLength,
                           const Matrix<typename H::ValueType, 2, 1, false>& offset) {
    H mat{0};
    mat.at(0u, 0u) = focalLength;
    mat.at(1u, 1u) = focalLength;
    mat.at(2u, 2u) = 1;
    mat.at(0u, 2u) = offset.at(0u, 0u);
    mat.at(1u, 2u) = offset.at(1u, 0u);
    return Camera<H>(mat);
}

// TODO page 170

// TODO depth page 180
