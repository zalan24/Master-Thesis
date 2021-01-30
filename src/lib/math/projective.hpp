#pragma once

#include "linearalgebra.hpp"
#include "matrix.hpp"

template <typename H>
class Projection
{
    static_assert(H::rows() == H::columns(), "The projection matrix has to be a square matrix");

 public:
    Projection() = default;
    explicit Projection(const H& h) : data(h), inv(::inverse(h)) {}

    operator H() const { return data; }

    const H& matrix() const { return data; }
    const H& inverseMatrix() const { return inv; }
    Projection inverse() const { return Projection(inv, data); }

    bool operator==(const Projection& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Projection& other) const { return !(*this == other); }

    Projection& operator*=(const Projection& other) {
        data *= other.data;
        // TODO this might cause a problem if transformation are applied repeatedly
        inv = other.inv * inv;
        return *this;
    }

    friend Projection operator*(const Projection& a, const Projection& b) {
        Projection ret = a;
        ret *= b;
        return ret;
    }

 private:
    H data;
    H inv;

    explicit Projection(const H& h, const H& i) : data(h), inv(i) {}
};

template <typename H, typename IH>
class Point
{
    static_assert(H::rows() == IH::rows() + 1,
                  "The homogeneous representation needs exactly one extra row");
    static_assert(H::columns() == 1, "The homogeneous representation needs exactly 1 column");
    static_assert(IH::columns() == 1, "The inhomogeneous representation needs exactly 1 column");

 public:
    using HMatType = H;
    using IHMatType = IH;
    Point() { homogeneousComponent() = 1; }
    explicit Point(const H& h) : data(h) {}
    explicit Point(const IH& ih, const typename H::ValueType& w) {
        for (unsigned int r = 0; r < ih.rows(); ++r)
            data.at(r, 0u) = ih.at(r, 0u);
        data.at(data.rows() - 1, 0u) = w;
    }

    const typename H::ValueType& homogeneousComponent() const {
        return data.at(data.rows() - 1, 0u);
    }
    typename H::ValueType& homogeneousComponent() { return data.at(data.rows() - 1, 0u); }

    explicit operator IH() const {
        ASSERT(homogeneousComponent() != 0);
        return data.template submatrix<H::rows() - 1, H::columns()>() / homogeneousComponent();
    }

    explicit operator H() const { return data; }
    const H& matrix() const { return data; }
    H& matrix() { return data; }
    IH inhomogeneous() const { return static_cast<IH>(*this); }

    bool operator==(const Point& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Point& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = static_cast<HH>(h) * data;
    }

    template <typename HH>
    Point transformed(const Projection<HH>& h) const {
        Point<H, IH> ret = *this;
        ret.transform(h);
        return ret;
    }

 private:
    H data;
};
