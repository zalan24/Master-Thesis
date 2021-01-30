#pragma once

#include "matrix.hpp"
#include "projective.hpp"
#include "types.hpp"

template <typename H = Mat33>
struct Projection2D_Helper
{
    static_assert(H::rows() == 3 && H::columns() == 3, "Projection2D has to be 3x3");
    using type = Projection<H>;
};

template <typename H = Mat33>
using Projection2D = typename Projection2D_Helper<H>::type;

template <typename H = Vec3, typename IH = Vec2>
class Point2D final : public Point<H, IH>
{
    static_assert(H::rows() == 3, "The homogeneous representation needs exactly 3 rows");

 public:
    Point2D() = default;
    explicit Point2D(const H& h) : Point<H, IH>(h) {}
    explicit Point2D(const IH& ih, const typename H::ValueType& w) : Point<H, IH>(ih, w) {}

    template <typename HH>
    Point2D transformed(const Projection<HH>& h) const {
        Point2D<H, IH> ret = *this;
        ret.transform(h);
        return ret;
    }
};

template <typename H = Vec3>
class Line2D final
{
    static_assert(H::columns() == 1, "The homogeneous representation needs exactly 1 column");
    static_assert(H::rows() == 3, "The homogeneous representation needs exactly 3 rows");

 public:
    using MatType = H;
    // using V2 = decltype(std::declval<H&>().template submatrix<H::rows() - 1, 1>());
    Line2D() : data({{1, 0, 0}}) {}
    template <typename V2,
              typename = std::enable_if_t<(V2::rows() == H::rows() - 1 && V2::columns() == 1)>>
    explicit Line2D(const V2& normal, const typename H::ValueType& dist) {
        for (unsigned int i = 0; i < V2::rows(); ++i)
            data.at(i, 0u) = normal.at(i, 0u);
        data.at(data.rows() - 1, 0u) = -dist;
    }
    explicit Line2D(const H& h) : data(h) {}

    explicit operator H() const { return data; }
    const H& matrix() const { return data; }
    H& matrix() { return data; }

    bool operator==(const Line2D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Line2D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = h.inverseMatrix().transpose() * data;
    }

    template <typename HH>
    Line2D transformed(const Projection<HH>& h) const {
        Line2D ret = *this;
        ret.transform(h);
        return ret;
    }

    auto normal() const { return data.template submatrix<2, 1>(); }

 private:
    H data;
};

template <typename H = Mat33>
class Conic2D final
{
 public:
    using ValueType = typename H::ValueType;
    using MatType = H;
    Conic2D() : data({{1, 0, 0}, {0, 1, 0}, {0, 0, -1}}) {}
    explicit Conic2D(const H& mat) : data(mat) {}
    explicit Conic2D(H&& mat) : data(std::move(mat)) {}
    explicit Conic2D(const ValueType& a, const ValueType& b, const ValueType& c, const ValueType& d,
                     const ValueType& e, const ValueType& f)
      : data({{a, b / static_cast<ValueType>(2), d / static_cast<ValueType>(2)},
              {b / static_cast<ValueType>(2), c, e / static_cast<ValueType>(2)},
              {d / static_cast<ValueType>(2), e / static_cast<ValueType>(2), f}}) {}

    operator H() const { return data; }
    const H& matrix() const { return data; }

    bool operator==(const Conic2D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Conic2D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = h.inverseMatrix().transpose() * data * h.inverseMatrix();
    }

    template <typename HH>
    Conic2D transformed(const Projection<HH>& h) const {
        Conic2D ret = *this;
        ret.transform(h);
        return ret;
    }

 private:
    H data;
};

template <typename H = Mat33>
class DualConic2D final
{
 public:
    using MatType = H;
    using ValueType = typename H::ValueType;
    DualConic2D() : data(inverse(H{{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}}})) {}
    template <typename HH>
    explicit DualConic2D(const Conic2D<HH>& c) : data(inverse(static_cast<HH>(c))) {}

    operator H() const { return data; }
    const H& matrix() const { return data; }

    bool operator==(const DualConic2D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const DualConic2D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = static_cast<HH>(h) * data * static_cast<HH>(h).transpose();
    }

    template <typename HH>
    DualConic2D transformed(const Projection<HH>& h) const {
        DualConic2D ret = *this;
        ret.transform(h);
        return ret;
    }

    explicit operator Conic2D<H>() const { return Conic2D<H>(inverse(data)); }

 private:
    H data;
};

template <typename H, typename IH>
Line2D<H> join(const Point2D<H, IH>& a, const Point2D<H, IH>& b) {
    return Line2D<H>(cross(static_cast<H>(a), static_cast<H>(b)));
}

template <typename H, typename IH = decltype(std::declval<H>().template submatrix<2, 1>())>
Point2D<H, IH> intersection(const Line2D<H>& a, const Line2D<H>& b) {
    return Point2D<H, IH>(cross(static_cast<H>(a), static_cast<H>(b)));
}

/**
 * Returns 0 if the point is on the line
 * The returned value is small if the point is near the line
 */
template <typename H, typename IH, typename LH>
typename H::ValueType match(const Point2D<H, IH>& p, const Line2D<LH>& l) {
    return dot(static_cast<LH>(l), static_cast<H>(p));
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const Line2D<LH>& l, const Point2D<H, IH>& p) {
    return match(p, l);
}

template <typename H, typename IH, typename LH>
auto match(const Point2D<H, IH>& p, const Conic2D<LH>& c) {
    using RetType = std::decay_t<decltype(static_cast<H>(p).transpose() * static_cast<LH>(c)
                                          * static_cast<H>(p))>;
    static_assert(RetType::rows() == 1 && RetType::columns() == 1,
                  "Bad return type in match(point, conic)");
    return determinant(static_cast<H>(p).transpose() * static_cast<LH>(c) * static_cast<H>(p));
}

template <typename H, typename IH, typename LH>
auto match(const Conic2D<LH>& c, const Point2D<H, IH>& p) {
    return match(p, c);
}

template <typename H, typename IH, typename HH>
Line2D<H> pole(const Point2D<H, IH>& point, const Conic2D<HH>& c) {
    return Line2D<H>(static_cast<HH>(c) * static_cast<H>(point));
}

// TODO: create Conic2D from 5 points (page 48)

// TODO: create circle from 3 points

// TODO: projective angles (page 72)

// TODO: recovery of metric propertion (page 73)
