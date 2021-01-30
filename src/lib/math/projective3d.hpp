#pragma once

#include "cameras.hpp"
#include "matrix.hpp"
#include "projective.hpp"
#include "types.hpp"

template <typename H = Mat33>
struct Projection3D_Helper
{
    static_assert(H::rows() == 4 && H::columns() == 4, "Projection3D has to be 4x4");
    using type = Projection<H>;
};

template <typename H = Mat44>
using Projection3D = typename Projection3D_Helper<H>::type;

template <typename H = Vec4, typename IH = Vec3>
class Point3D final : public Point<H, IH>
{
    static_assert(H::rows() == 4, "The homogeneous representation needs exactly 4 rows");

 public:
    Point3D() = default;
    explicit Point3D(const H& h) : Point<H, IH>(h) {}
    explicit Point3D(const IH& ih, const typename H::ValueType& w) : Point<H, IH>(ih, w) {}

    template <typename HH>
    Point3D transformed(const Projection<HH>& h) const {
        Point3D<H, IH> ret = *this;
        ret.transform(h);
        return ret;
    }

    template <typename HH>
    auto transformed(const Camera<HH>& h) const {
        Point2D<Matrix<typename H::ValueType, 3u, 1u, H::isDynamic()>,
                Matrix<typename IH::ValueType, 2u, 1u, IH::isDynamic()>>
          ret{h.matrix() * Point<H, IH>::matrix()};
        return ret;
    }
};

template <typename H = Vec4>
class Plane3D final
{
    static_assert(H::columns() == 1, "The homogeneous representation needs exactly 1 column");
    static_assert(H::rows() == 4, "The homogeneous representation needs exactly 4 rows");

 public:
    using MatType = H;
    Plane3D() : data({{1, 0, 0, 0}}) {}
    template <typename V3,
              typename = std::enable_if_t<(V3::rows() == H::rows() - 1 && V3::columns() == 1)>>
    explicit Plane3D(const V3& normal, const typename H::ValueType& dist) {
        for (unsigned int i = 0; i < V3::rows(); ++i)
            data.at(i, 0u) = normal.at(i, 0u);
        data.at(data.rows() - 1, 0u) = -dist;
    }
    explicit Plane3D(const H& h) : data(h) {}

    explicit operator H() const { return data; }
    const H& matrix() const { return data; }

    bool operator==(const Plane3D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Plane3D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = h.inverseMatrix().transpose() * data;
    }

    template <typename HH>
    Plane3D transformed(const Projection<HH>& h) const {
        Plane3D ret = *this;
        ret.transform(h);
        return ret;
    }

 private:
    H data;
};

template <typename H = Mat44>
class Line3D final
{
    static_assert(H::columns() == 4, "The homogeneous representation needs exactly 4 column");
    static_assert(H::rows() == 4, "The homogeneous representation needs exactly 4 rows");

 public:
    using MatType = H;
    using CoordinatesMat = Matrix<typename H::ValueType, 6, 1, H::isDynamic()>;
    Line3D() : data({{0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, 0}}) {}
    explicit Line3D(const H& h) : data(h) {}
    explicit Line3D(const CoordinatesMat& c)
      : data({{0, c.at(0u, 0u), c.at(1u, 0u), c.at(2u, 0u)},
              {-c.at(0u, 0u), 0, c.at(3u, 0u), c.at(4u, 0u)},
              {-c.at(1u, 0u), -c.at(3u, 0u), 0, c.at(5u, 0u)},
              {-c.at(2u, 0u), -c.at(4u, 0u), -c.at(5u, 0u), 0}}) {}
    template <typename HH, typename IH, typename HH2, typename IH2>
    explicit Line3D(const Point3D<HH, IH>& a, const Point3D<HH2, IH2>& b)
      : data(a.matrix() * b.matrix().transpose() - b.matrix() * a.matrix().transpose()) {}

    explicit operator H() const { return data; }
    const H& matrix() const { return data; }

    bool operator==(const Line3D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const Line3D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = h.matrix() * data * h.matrix().transpose();
    }

    template <typename HH>
    Line3D transformed(const Projection<HH>& h) const {
        Line3D ret = *this;
        ret.transform(h);
        return ret;
    }

    template <typename HH, bool D = false>
    auto transformed(const Camera<HH>& h) const {
        // https://math.stackexchange.com/questions/1811665/how-to-project-a-3d-line-represented-in-pl%C3%BCcker-coordinates-into-2d-image
        const auto lx = h.matrix() * data * h.matrix().transpose();
        using V = Matrix<typename H::ValueType, 3u, 1u, D>;
        Line2D<V> ret{V{{{lx.at(2u, 1u), lx.at(0u, 2u), lx.at(1u, 0u)}}}};
        return ret;
    }

    CoordinatesMat coordinates() const {
        return CoordinatesMat({{data.at(1u, 0u), data.at(2u, 0u), data.at(3u, 0u), data.at(2u, 1u),
                                data.at(3u, 1u), data.at(3u, 2u)}});
    }

 private:
    H data;
};

template <typename H = Mat44>
class DualLine3D final
{
    static_assert(H::columns() == 4, "The homogeneous representation needs exactly 4 column");
    static_assert(H::rows() == 4, "The homogeneous representation needs exactly 4 rows");

 public:
    using MatType = H;
    using CoordinatesMat = Matrix<typename H::ValueType, 6, 1, H::isDynamic()>;
    DualLine3D() : data({{0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, 0}}) {}
    explicit DualLine3D(const H& h) : data(h) {}
    explicit DualLine3D(const CoordinatesMat& c)
      : data({{0, c.at(0u, 0u), c.at(1u, 0u), c.at(2u, 0u)},
              {-c.at(0u, 0u), 0, c.at(3u, 0u), c.at(4u, 0u)},
              {-c.at(1u, 0u), -c.at(3u, 0u), 0, c.at(5u, 0u)},
              {-c.at(2u, 0u), -c.at(4u, 0u), -c.at(5u, 0u), 0}}) {}
    explicit DualLine3D(const Line3D<H>& line)
      : DualLine3D(reverseCoordinates<CoordinatesMat>(line.coordinates())) {}
    template <typename HH, typename HH2>
    explicit DualLine3D(const Plane3D<HH>& a, const Plane3D<HH2>& b)
      : data(a.matrix() * b.matrix().transpose() - b.matrix() * a.matrix().transpose()) {}

    explicit operator H() const { return data; }
    const H& matrix() const { return data; }

    bool operator==(const DualLine3D& other) const { return scaledEqual(data, other.data); }
    bool operator!=(const DualLine3D& other) const { return !(*this == other); }

    template <typename HH>
    void transform(const Projection<HH>& h) {
        data = h.inverseMatrix().transpose() * data * h.inverseMatrix();
    }

    template <typename HH>
    DualLine3D transformed(const Projection<HH>& h) const {
        DualLine3D ret = *this;
        ret.transform(h);
        return ret;
    }

    CoordinatesMat coordinates() const {
        return CoordinatesMat({{data.at(1u, 0u), data.at(2u, 0u), data.at(3u, 0u), data.at(2u, 1u),
                                data.at(3u, 1u), data.at(3u, 2u)}});
    }

    explicit operator Line3D<H>() const {
        return Line3D<H>(reverseCoordinates<typename Line3D<H>::CoordinatesMat>(coordinates()));
    }
    Line3D<H> line() const { return static_cast<Line3D<H>>(*this); }

 private:
    template <typename M, typename M2>
    static M reverseCoordinates(const M2& coords) {
        return M({{coords.at(5u, 0u), -coords.at(4u, 0u), coords.at(3u, 0u), coords.at(2u, 0u),
                   -coords.at(1u, 0u), coords.at(0u, 0u)}});
    }
    H data;
};

/**
 * Returns 0 if the point is on the line
 * The returned value is small if the point is near the line
 */
template <typename H, typename IH, typename LH>
typename H::ValueType match(const Point3D<H, IH>& p, const Plane3D<LH>& l) {
    return dot(static_cast<LH>(l), static_cast<H>(p));
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const Plane3D<LH>& l, const Point3D<H, IH>& p) {
    return match(p, l);
}

template <typename H, typename IH>
Plane3D<H> join(const Point3D<H, IH>& a, const Point3D<H, IH>& b, const Point3D<H, IH>& c) {
    return Plane3D(cross(a.matrix(), b.matrix(), c.matrix()));
}

template <typename H, typename IH,
          typename LH = decltype(std::declval<H>() * std::declval<H>().transpose())>
Line3D<LH> join(const Point3D<H, IH>& a, const Point3D<H, IH>& b) {
    return Line3D<LH>(a, b);
}

template <typename H, typename IH = decltype(std::declval<H>().template submatrix<3, 1>())>
Point3D<H, IH> intersection(const Plane3D<H>& a, const Plane3D<H>& b, const Plane3D<H>& c) {
    return Point3D<H, IH>(cross(a.matrix(), b.matrix(), c.matrix()));
}

template <typename H, typename H2,
          typename LH = decltype(std::declval<H>() * std::declval<H2>().transpose())>
DualLine3D<LH> intersection(const Plane3D<H>& a, const Plane3D<H2>& b) {
    return DualLine3D<LH>(a, b);
}

template <typename H, typename IH, typename H2>
auto join(const Point3D<H, IH>& point, const DualLine3D<H2>& line) {
    return Plane3D<H>(line.matrix() * point.matrix());
}

template <typename H, typename IH, typename H2>
auto join(const DualLine3D<H2>& line, const Point3D<H, IH>& point) {
    return join(point, line);
}

template <typename H, typename IH, typename H2>
auto join(const Point3D<H, IH>& point, const Line3D<H2>& line) {
    return join(point, DualLine3D<H2>(line));
}

template <typename H, typename IH, typename H2>
auto join(const Line3D<H2>& line, const Point3D<H, IH>& point) {
    return join(point, DualLine3D<H2>(line));
}

template <typename H, typename H2,
          typename IH = decltype(std::declval<H>().template submatrix<3, 1>())>
Point3D<H, IH> intersection(const Line3D<H2>& line, const Plane3D<H>& plane) {
    return Point3D<H, IH>(line.matrix() * plane.matrix());
}

template <typename H, typename H2,
          typename IH = decltype(std::declval<H>().template submatrix<3, 1>())>
Point3D<H, IH> intersection(const Plane3D<H>& plane, const Line3D<H2>& line) {
    return intersection(line, plane);
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const Point3D<H, IH>& p, const DualLine3D<LH>& l) {
    return matrixSquareSum(join(p, l).matrix());
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const DualLine3D<LH>& l, const Point3D<H, IH>& p) {
    return match(p, l);
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const Point3D<H, IH>& p, const Line3D<LH>& l) {
    return match(p, DualLine3D<LH>(l));
}

template <typename H, typename IH, typename LH>
typename H::ValueType match(const Line3D<LH>& l, const Point3D<H, IH>& p) {
    return match(p, l);
}

template <typename H, typename LH>
typename H::ValueType match(const Plane3D<H>& p, const Line3D<LH>& l) {
    return matrixSquareSum(intersection(p, l).matrix());
}

template <typename H, typename LH>
typename H::ValueType match(const Line3D<LH>& l, const Plane3D<H>& p) {
    return match(p, l);
}

template <typename H, typename PH = Matrix<typename H::ValueType, 4, 1, H::isDynamic()>>
auto direction(const Line3D<H>& l) {
    return intersection(l, Plane3D<PH>(PH({{0, 0, 0, 1}})));
}

template <typename V, bool D1, bool D2, bool D3>
Matrix<V, 4, 4, D1> lookAt(const Matrix<V, 3, 1, D1>& eye, const Matrix<V, 3, 1, D2>& up,
                           const Matrix<V, 3, 1, D3>& target) {
    Matrix<V, 4, 4, D1> ret;
    const auto dir = normalize(target - eye);
    const auto side = normalize(cross(up, dir));
    const auto correctedUp = cross(dir, side);
    for (unsigned int r = 0; r < 3; ++r) {
        ret.at(r, 0u) = side.at(r, 0u);
        ret.at(r, 1u) = correctedUp.at(r, 0u);
        ret.at(r, 2u) = dir.at(r, 0u);
        ret.at(r, 3u) = eye.at(r, 0u);
    }
    return ret;
}

// TODO: parameterized points on a plane (page 86) ??? this is not easy

// TODO: quadrics (page 91)

// TODO: twisted cubics (page 93)

// TODO: plane angles (page 103)
