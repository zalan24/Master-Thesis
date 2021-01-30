#pragma once

#include <util.hpp>

#include "cameras.hpp"
#include "matrix.hpp"
#include "spline.hpp"

using PrecFloat = long double;
#ifdef DEBUG
using BaseFloat = double;
#else
using BaseFloat = float;
#endif
using DebugFloat = EFloat<BaseFloat, PrecFloat>;

#ifdef DEBUG
using Float = DebugFloat;
#else
using Float = BaseFloat;
#endif

static const std::size_t maxLocalBytes = 64;

template <typename F>
constexpr static bool shouldBeDynamic(unsigned int R, unsigned int C) {
    return R * C * sizeof(F) > maxLocalBytes;
}

template <typename F, unsigned int R, unsigned int C, bool D = shouldBeDynamic<F>(R, C)>
using MatrixT = Matrix<F, R, C, D>;

template <unsigned int R, unsigned int C>
using MatrixF = MatrixT<Float, R, C>;

template <unsigned int R, unsigned int C>
using MatrixI = MatrixT<int, R, C>;

template <unsigned int R, unsigned int C>
using MatrixU = MatrixT<unsigned int, R, C>;

using Mat22 = MatrixF<2, 2>;
using Mat33 = MatrixF<3, 3>;
using Mat44 = MatrixF<4, 4>;

using Vec2 = MatrixF<2, 1>;
using Vec3 = MatrixF<3, 1>;
using Vec4 = MatrixF<4, 1>;

using IVec2 = MatrixI<2, 1>;
using IVec3 = MatrixI<3, 1>;
using IVec4 = MatrixI<4, 1>;

using UVec2 = MatrixU<2, 1>;
using UVec3 = MatrixU<3, 1>;
using UVec4 = MatrixU<4, 1>;

#ifndef DEBUG
static_assert(!Vec2::isDynamic(), "Vec2 should not be dynamicly stored");
static_assert(!Vec3::isDynamic(), "Vec3 should not be dynamicly stored");
static_assert(!Vec4::isDynamic(), "Vec4 should not be dynamicly stored");
#endif

template <typename T>
using Interval = Range<T, 1>;

template <typename T>
using Rect = Range<T, 2>;

using IntervalF = Interval<Float>;
using IntervalI = Interval<int>;
using IntervalU = Interval<unsigned int>;
using RectF = Rect<Float>;
using RectI = Rect<int>;
using RectU = Rect<unsigned int>;

using CameraF = Camera<MatrixF<3, 4>>;

using SplineF = Spline<Float, Float>;
using SplineVec2 = Spline<Vec2, Float>;
using SplineVec3 = Spline<Vec3, Float>;
