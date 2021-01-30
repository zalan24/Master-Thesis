#pragma once

#include "matrix.hpp"
#include "types.hpp"

template <typename T, unsigned int R, bool D, bool D2>
T dot(const Matrix<T, R, 1, D>& a, const Matrix<T, R, 1, D2>& b) {
    using M = decltype(a.transpose() * b);
    static_assert(M::rows() == 1 && M::columns() == 1, "vT*v must be an 1x1 matrix");
    return (a.transpose() * b).at(0u, 0u);
}

template <typename T, unsigned int R, bool D>
T lengthSq(const Matrix<T, R, 1, D>& v) {
    return dot(v, v);
}

template <typename T, unsigned int R, bool D>
T length(const Matrix<T, R, 1, D>& v) {
    using math::sqrt;
    return sqrt(lengthSq(v));
}

template <typename T, unsigned int R, bool D, bool D2>
T distance(const Matrix<T, R, 1, D>& a, const Matrix<T, R, 1, D2>& b) {
    return length(a - b);
}

template <typename T, unsigned int R, bool D>
Matrix<T, R, 1, D> normalize(const Matrix<T, R, 1, D>& a) {
    return a / length(a);
}

template <bool... D>
struct Cross_helperD;

template <bool H, bool... D>
struct Cross_helperD<H, D...>
{ static const bool value = Cross_helperD<D...>::value && H; };

template <>
struct Cross_helperD<>
{ static const bool value = true; };

template <typename T, unsigned int R, bool DD, unsigned int I, bool... D>
struct Cross_helperM;

template <typename T, unsigned int R, bool DD, unsigned int I>
struct Cross_helperM<T, R, DD, I>
{
    static void combine(Matrix<T, R - 1, R, DD>&) {}
};

template <typename T, unsigned int R, bool DD, unsigned int I, bool H, bool... D>
struct Cross_helperM<T, R, DD, I, H, D...>
{
    static void combine(Matrix<T, R - 1, R, DD>& target, const Matrix<T, R, 1, H>& a,
                        const Matrix<T, R, 1, D>&... as) {
        for (unsigned int i = 0; i < R; ++i)
            target.at(I, i) = a.at(i, 0u);
        Cross_helperM<T, R, DD, I + 1, D...>::combine(target, as...);
    }
};

template <typename T, unsigned int R, bool... D>
auto cross(const Matrix<T, R, 1, D>&... a) {
    static_assert(sizeof...(D) == R - 1, "Number of arguments for cross product must be R-1");
    Matrix<T, R - 1, R, Cross_helperD<D...>::value> mat;
    Cross_helperM<T, R, Cross_helperD<D...>::value, 0, D...>::combine(mat, a...);

    Matrix<T, R, 1, Cross_helperD<D...>::value> ret(0);
    for (unsigned int i = 0; i < R; ++i)
        ret.at(i, 0u) = determinant(removeColumn(mat, i)) * static_cast<T>((i & 1) ? -1 : 1);
    return ret;
}

template <typename M = Mat44, typename V, typename F>
M rotate(const V& axis, const F& angle) {
    static_assert(M::rows() >= 3 && M::columns() >= 3, "Bad matrix size for rotation");
    static_assert(V::rows() == 3 && V::columns() == 1,
                  "Bad matrix dimensions for transaltion vector");
    M ret;
    using math::cos;
    using math::sin;
    const auto& x = axis.at(0u, 0u);
    const auto& y = axis.at(1u, 0u);
    const auto& z = axis.at(2u, 0u);
    const auto cs = static_cast<typename M::ValueType>(cos(angle));
    const auto sn = static_cast<typename M::ValueType>(sin(angle));
    const auto one = static_cast<typename M::ValueType>(1);
    ret.at(0u, 0u) = cs + x * x * (one - cs);
    ret.at(1u, 0u) = y * x * (one - cs) + z * sn;
    ret.at(2u, 0u) = z * x * (one - cs) - y * sn;

    ret.at(0u, 1u) = y * x * (one - cs) - z * sn;
    ret.at(1u, 1u) = cs + y * y * (one - cs);
    ret.at(2u, 1u) = z * y * (one - cs) + x * sn;

    ret.at(0u, 2u) = z * x * (one - cs) + y * sn;
    ret.at(1u, 2u) = z * y * (one - cs) - x * sn;
    ret.at(2u, 2u) = cs + z * z * (one - cs);
    return ret;
}

template <typename M = Mat44, typename T>
M translate(const T& translation) {
    M ret;
    static_assert(M::rows() == M::columns(), "Wrong matrix type");
    static_assert(T::rows() + 1 == M::rows() && T::columns() == 1,
                  "Bad matrix dimensions for transaltion vector");
    for (unsigned int i = 0; i < T::rows(); ++i)
        ret.at(i, M::columns() - 1) = translation.at(i, 0u);
    return ret;
}

template <typename M = Mat44, typename V>
M diag(const V& d) {
    static_assert(M::rows() == M::columns(), "A diagonal matrix has no be square matrix");
    static_assert(V::columns() == 1, "Expected a column vector");
    static_assert(V::rows() <= M::rows(), "The diagonal vector cannot be larger than the matrix");
    M ret;
    for (unsigned int i = 0; i < V::rows(); ++i)
        ret.at(i, i) = d.at(i, 0u);
    return ret;
}

template <typename M = Mat44, typename F>
M scale(const F& s) {
    static_assert(M::rows() == M::columns() && M::rows() > 1, "Wrong matrix type");
    Matrix<typename M::ValueType, M::rows() - 1, 1, false> d;
    foreachElement(d, [&](auto& val) { val = s; });
    return diag<M>(d);
}
