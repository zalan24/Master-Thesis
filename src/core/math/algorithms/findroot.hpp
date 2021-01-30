#pragma once

#include <util.hpp>

//! solve a*x + b = 0
template <typename T>
void find_roots(const T& a, const T& b, T& x) {
    x = -b / a;
}

//! solve a*x^2 + b*x + c = 0
template <typename T>
void find_roots(const T& a, const T& b, const T& c, T& x1, T& x2) {
    T q = b * b - T{4} * a * c;
    using math::sqrt;
    q = sqrt(q);
    T _2a = T{1} / (T{2} * a);
    x1 = (-b + q) * _2a;
    x2 = (-b - q) * _2a;
}

//! solve a*x^2 + b*x + c = 0
template <typename T>
void find_roots_forced(const T& a, const T& b, const T& c, T& x1, T& x2) {
    T q = b * b - T{4} * a * c;
    using math::sqrt;
    q = q < T{0} ? T{0} : sqrt(q);
    T _2a = T{1} / (T{2} * a);
    x1 = (-b + q) * _2a;
    x2 = (-b - q) * _2a;
}

//! solve t^3 + p*t + q = 0
template <typename T>
void find_roots_depressed_cubic(const T& p, const T& q, T& t1, T& t2, T& t3) {
    using math::asin;
    using math::cbrt;
    using math::cos;
    using math::nan;
    using math::sin;
    using math::sqrt;
    T d = T{27} * q * q + T{4} * p * p * p;
    if (d < T{0}) {
        // https://stackoverflow.com/questions/2003465/fastest-numerical-solution-of-a-real-cubic-polynomial
        // const T sqrt_p = sqrt(-p);
        // const T _2_s3 = T{2} / sqrt(T{3});
        // T v = sqrt(T{3}) * T{3} * q / (T{2} * sqrt_p * sqrt_p * sqrt_p);
        // v = asin(v) / T{3};
        // t1 = _2_s3 * sqrt_p * sin(v) - a / T{3};
        // t2 = -_2_s3 * sqrt_p * sin(v + T{M_PI} / T{3}) - a / T{3};
        // t3 = _2_s3 * sqrt_p * cos(v + T{M_PI} / T{6}) - a / T{3};

        const T sqrt_p_3 = sqrt(-p / T{3});
        const T ac = acos(T{3} * q / (T{2} * p * sqrt_p_3)) / T{3};
        const T m = T{2} * sqrt_p_3;
        t1 = m * cos(ac);
        t2 = m * cos(ac - T{2} * T{M_PI} / T{3});
        t3 = m * cos(ac - T{4} * T{M_PI} / T{3});
    }
    else if (d > T{0}) {
        d = sqrt(d);
        t1 = cbrt(-q / T{2} + d) + cbrt(-q / T{2} - d);
        nan(t1);
        nan(t2);
    }
    else {  // d == 0
        t1 = T{-2} * cbrt(q / T{2});
        t2 = t3 = cbrt(q / T{2});
    }
}

//! solve a*x^3 + b*x^2 + c*x + d = 0
template <typename T>
void find_roots(const T& a, const T& b, const T& c, const T& d, T& x1, T& x2, T& x3) {
    const T invA = T{1} / a;
    const T b_3a = b * invA / T{3};
    const T p = (T{3} * a * c - b * b) * invA * invA / T{3};
    const T q =
      (T{2} * b * b * b - T{9} * a * b * c + T{27} * a * a * d) * invA * invA * invA / T{27};
    ::find_roots_depressed_cubic(p, q, x1, x2, x3);
    x1 -= b_3a;
    x2 -= b_3a;
    x3 -= b_3a;
}
