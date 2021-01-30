#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include "matrixref.hpp"

template <typename T, unsigned int R, unsigned int C>
struct MatrixStorage_Dynamic
{
    std::vector<T> data;
    MatrixStorage_Dynamic() : data(R * C) {}
    template <typename I>
    T& operator[](I i) {
        return data[i];
    }
    template <typename I>
    const T& operator[](I i) const {
        return data[i];
    }
};

template <typename T, unsigned int R, unsigned int C>
using MatrixStorage_Static = T[R * C];

template <typename T, unsigned int R, unsigned int C, bool D>
struct MatrixStorageHelper;

template <typename T, unsigned int R, unsigned int C>
struct MatrixStorageHelper<T, R, C, true>
{ using type = MatrixStorage_Dynamic<T, R, C>; };

template <typename T, unsigned int R, unsigned int C>
struct MatrixStorageHelper<T, R, C, false>
{ using type = MatrixStorage_Static<T, R, C>; };

template <typename T, unsigned int R, unsigned int C, bool D>
class alignas(16) Matrix final
{
    static_assert(R > 0 && C > 0, "Matrix size cannot be 0");

 public:
    static constexpr bool isDynamic() { return D; }
    using ValueType = T;
    explicit Matrix(const T& scale = 1) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) = r == c ? scale : 0;
    }

    explicit Matrix(const std::vector<std::vector<T>>& columns) {
        assert(columns.size() == C);
        for (std::size_t c = 0; c < C; ++c) {
            assert(columns[c].size() == R);
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) = columns[c][r];
        }
    }

    template <unsigned int RR, unsigned int CC, bool DD = D, typename I = unsigned int>
    Matrix<T, RR, CC, DD> submatrix(const I& rOffset = 0, const I& cOffset = 0) const {
        static_assert(RR <= R && CC <= C, "Submatrix has to be smaller than the original matrix");
        ASSERT(rOffset <= R - RR && cOffset <= C - CC);
        Matrix<T, RR, CC, DD> ret;
        for (std::size_t c = 0; c < CC; ++c)
            for (std::size_t r = 0; r < RR; ++r)
                ret.at(r, c) = at(r + rOffset, c + cOffset);
        return ret;
    }

    template <typename TT, bool DD = D>
    explicit operator Matrix<TT, R, C, DD>() const {
        Matrix<TT, R, C, DD> ret;
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                ret.at(r, c) = static_cast<TT>(at(r, c));
        return ret;
    }

    template <typename TT>
    Matrix<TT, R, C, D> castTo() const {
        return static_cast<Matrix<TT, R, C, D>>(*this);
    }

    template <bool DD>
    operator Matrix<T, R, C, DD>() const {
        Matrix<T, R, C, DD> ret;
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                ret.at(r, c) = at(r, c);
        return ret;
    }

    static constexpr unsigned int rows() { return R; }
    static constexpr unsigned int columns() { return C; }

    template <typename I>
    T& at(I r, I c) {
        static_assert(std::is_integral_v<I>,
                      "It is only possible to index a matrix with an integral");
        ASSERT(r < R && r >= 0 && c < C && c >= 0);
        return data[r + c * R];
    }
    template <typename I>
    const T& at(I r, I c) const {
        static_assert(std::is_integral_v<I>,
                      "It is only possible to index a matrix with an integral");
        ASSERT(r < R && r >= 0 && c < C && c >= 0);
        return data[r + c * R];
    }

    template <bool DD>
    Matrix<T, C, R, DD>& transpose(Matrix<T, C, R, DD>& target) const {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                target.at(c, r) = at(r, c);
        return target;
    }

    Matrix<T, C, R, D> transpose() const {
        Matrix<T, C, R, D> ret;
        transpose(ret);
        return ret;
    }

    template <bool DD>
    Matrix& operator+=(const Matrix<T, R, C, DD>& other) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) += other.at(r, c);
        return *this;
    }

    template <bool DD>
    Matrix& operator-=(const Matrix<T, R, C, DD>& other) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) -= other.at(r, c);
        return *this;
    }

    Matrix& operator*=(const T& s) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) *= s;
        return *this;
    }

    Matrix& operator/=(const T& s) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                at(r, c) /= s;
        return *this;
    }

    template <bool DD>
    Matrix<T, R, C, DD>& negate(Matrix<T, R, C, DD>& target) const {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                target.at(r, c) = -at(r, c);
        return target;
    }

    friend Matrix operator-(const Matrix& mat) {
        Matrix ret;
        mat.negate(ret);
        return ret;
    }

    friend Matrix operator+(const Matrix& a, const Matrix& b) {
        Matrix ret = a;
        ret += b;
        return ret;
    }

    friend Matrix operator-(const Matrix& a, const Matrix& b) {
        Matrix ret = a;
        ret -= b;
        return ret;
    }

    friend Matrix operator*(const Matrix& a, const T& s) {
        Matrix ret = a;
        ret *= s;
        return ret;
    }

    friend Matrix operator*(const T& s, const Matrix& a) {
        Matrix ret = a;
        ret *= s;
        return ret;
    }

    friend Matrix operator/(const Matrix& a, const T& s) {
        Matrix ret = a;
        ret /= s;
        return ret;
    }

    template <bool DD>
    friend Matrix elementwiseMul(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        Matrix ret;
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                ret.at(r, c) = a.at(r, c) * b.at(r, c);
        return ret;
    }

    template <bool DD>
    friend Matrix elementwiseDiv(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        Matrix ret;
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                ret.at(r, c) = a.at(r, c) / b.at(r, c);
        return ret;
    }

    template <unsigned int CC, bool DD>
    friend Matrix<T, R, CC, D> operator*(const Matrix& a, const Matrix<T, C, CC, DD>& b) {
        Matrix<T, R, CC, D> ret(0);
        for (std::size_t c = 0; c < CC; ++c) {
            for (std::size_t r = 0; r < R; ++r) {
                for (std::size_t k = 0; k < C; ++k)
                    ret.at(r, c) += a.at(r, k) * b.at(k, c);
            }
        }
        return ret;
    }

    Matrix& operator*=(const Matrix& other) {
        static_assert(R == C, "Only square matrices can use the *= operator with an other matrix");
        Matrix result = (*this) * other;
        (*this) = result;
        return (*this);
    }

    template <bool DD>
    friend ValueType scaledDiff(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        using std::abs;
        unsigned int refR = 0u;
        unsigned int refC = 0u;
        auto valAt = [&](unsigned int x, unsigned int y) {
            return std::min(abs(a.at(x, y)), abs(b.at(x, y)));
        };
        for (unsigned int c = 0u; c < C; ++c) {
            for (unsigned int r = 0u; r < R; ++r) {
                if (valAt(r, c) > valAt(refR, refC)) {
                    refR = r;
                    refC = c;
                }
            }
        }
        ValueType m = a.at(refR, refC);
        ValueType m2 = b.at(refR, refC);
        if (m == 0 || m2 == 0)
            m = m2 = 1;
        ValueType ret = 0;
        for (unsigned int c = 0; c < C; ++c) {
            for (unsigned int r = 0; r < R; ++r) {
                const auto val = a.at(r, c) * m2 - b.at(r, c) * m;
                ret += val * val;
            }
        }
        return ret;
    }

    template <bool DD>
    friend bool scaledEqual(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        return scaledDiff(a, b) == 0;
    }

    template <bool DD>
    friend bool operator==(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t r = 0; r < R; ++r)
                if (a.at(r, c) != b.at(r, c))
                    return false;
        return true;
    }
    template <bool DD>
    friend bool operator!=(const Matrix& a, const Matrix<T, R, C, DD>& b) {
        return !(a == b);
    }

    friend T error(const Matrix& a) { return ::error(a.cref()); }

    ValueType maxAbsValuedComponent() const {
        using std::abs;
        ValueType ret = at(0u, 0u);
        for (unsigned int c = 0; c < columns(); ++c)
            for (unsigned int r = 0; r < rows(); ++r)
                if (abs(ret) < abs(at(r, c)))
                    ret = at(r, c);
        return ret;
    }

    MatrixRef<T, ColumnMajorIndexing> ref() {
        return MatrixRef<T, ColumnMajorIndexing>(&data[0u], R, C);
    }

    ConstMatrixRef<T, ColumnMajorIndexing> cref() const {
        return ConstMatrixRef<T, ColumnMajorIndexing>(&data[0u], R, C);
    }

    MatrixRef<T, RowMajorIndexing> transpose_ref() {
        return MatrixRef<T, RowMajorIndexing>(&data[0u], R, C);
    }

    ConstMatrixRef<T, RowMajorIndexing> transpose_cref() const {
        return ConstMatrixRef<T, RowMajorIndexing>(&data[0u], R, C);
    }

 private:
    using MatrixStorage_t = typename MatrixStorageHelper<T, R, C, D>::type;
    MatrixStorage_t data;
};

template <typename T, unsigned int R, unsigned int C, bool D, typename I>
auto removeColumn(const Matrix<T, R, C, D>& m, const I& column) {
    static_assert(C > 1, "The size of the resulting matrix would become 0 in a dimension");
    assert(column >= 0 && column < C);
    Matrix<T, R, C - 1, D> ret;
    for (std::size_t c = 0; c < C; ++c) {
        if (c == column)
            continue;
        std::size_t cc = c > column ? c - 1 : c;
        for (std::size_t r = 0; r < R; ++r)
            ret.at(r, cc) = m.at(r, c);
    }
    return ret;
}

template <typename T, unsigned int R, bool D1, bool D2>
Matrix<T, R, 1u, D1> operator*(const Matrix<T, R, 1u, D1>& lhs, const Matrix<T, R, 1u, D2>& rhs) {
    Matrix<T, R, 1u, D1> ret;
    for (unsigned int r = 0; r < R; ++r)
        ret.at(r, 0u) = lhs.at(r, 0u) * rhs.at(r, 0u);
    return ret;
}

template <typename T, unsigned int R, unsigned int C, bool D, typename I>
auto removeRow(const Matrix<T, R, C, D>& m, const I& row) {
    static_assert(R > 1, "The size of the resulting matrix would become 0 in a dimension");
    assert(row >= 0 && row < R);
    Matrix<T, R - 1, C, D> ret;
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t r = 0; r < R; ++r) {
            if (r == row)
                continue;
            std::size_t rr = r > row ? r - 1 : r;
            ret.at(rr, c) = m.at(r, c);
        }
    }
    return ret;
}

template <typename T, unsigned int R, unsigned int C, bool D, typename I>
auto removeRowAndColumn(const Matrix<T, R, C, D>& m, const I& row, const I& column) {
    static_assert(R > 1 && C > 1, "The size of the resulting matrix would become 0 in a dimension");
    assert(row >= 0 && row < R && column >= 0 && column < C);
    Matrix<T, R - 1, C - 1, D> ret;
    for (std::size_t c = 0; c < C; ++c) {
        if (c == column)
            continue;
        std::size_t cc = c > column ? c - 1 : c;
        for (std::size_t r = 0; r < R; ++r) {
            if (r == row)
                continue;
            std::size_t rr = r > row ? r - 1 : r;
            ret.at(rr, cc) = m.at(r, c);
        }
    }
    return ret;
}

template <typename T, bool D>
T determinant(const Matrix<T, 1, 1, D>& m) {
    return m.at(0u, 0u);
}

template <typename T, bool D>
T determinant(const Matrix<T, 2, 2, D>& m) {
    return m.at(0u, 0u) * m.at(1u, 1u) - m.at(1u, 0u) * m.at(0u, 1u);
}

template <typename T, unsigned int S, bool D>
T determinant(const Matrix<T, S, S, D>& m) {
    T sum = 0;
    for (std::size_t c = 0; c < S; ++c)
        sum += m.at(static_cast<std::size_t>(0), c)
               * determinant(removeRowAndColumn(m, static_cast<std::size_t>(0), c))
               * static_cast<T>((c & 1) ? -1 : 1);
    return sum;
}

template <typename T, bool D>
Matrix<T, 1, 1, D> inverse(const Matrix<T, 1, 1, D>& m) {
    return Matrix<T, 1, 1, D>(static_cast<T>(1) / determinant(m));
}

template <typename T, bool D>
Matrix<T, 2, 2, D> inverse(const Matrix<T, 2, 2, D>& m) {
    return Matrix<T, 2, 2, D>({{m.at(1u, 1u), -m.at(1u, 0u)}, {-m.at(0u, 1u), m.at(0u, 0u)}})
           / determinant(m);
}

template <typename T, unsigned int R1, unsigned int R2, unsigned int C1, unsigned int C2, bool D1,
          bool D2, bool D3, bool D4, bool DD = D1>
Matrix<T, R1 + R2, C1 + C2, DD> combine(const Matrix<T, R1, C1, D1>& A,
                                        const Matrix<T, R1, C2, D2>& B,
                                        const Matrix<T, R2, C1, D3>& C,
                                        const Matrix<T, R2, C2, D4>& D) {
    Matrix<T, R1 + R2, C1 + C2, DD> ret;
    for (std::size_t c = 0; c < A.columns(); ++c)
        for (std::size_t r = 0; r < A.rows(); ++r)
            ret.at(r, c) = A.at(r, c);
    for (std::size_t c = 0; c < B.columns(); ++c)
        for (std::size_t r = 0; r < B.rows(); ++r)
            ret.at(r, c + A.columns()) = B.at(r, c);
    for (std::size_t c = 0; c < C.columns(); ++c)
        for (std::size_t r = 0; r < C.rows(); ++r)
            ret.at(r + A.rows(), c) = C.at(r, c);
    for (std::size_t c = 0; c < D.columns(); ++c)
        for (std::size_t r = 0; r < D.rows(); ++r)
            ret.at(r + A.rows(), c + A.columns()) = D.at(r, c);
    return ret;
}

template <typename T, unsigned int R, unsigned int C, bool D, typename F>
void foreachElement(Matrix<T, R, C, D>& m, F&& f) {
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t r = 0; r < R; ++r)
            f(m.at(r, c));
}

template <typename T, unsigned int R, unsigned int C, bool D, typename F>
void foreachElement(const Matrix<T, R, C, D>& m, F&& f) {
    for (std::size_t c = 0; c < C; ++c)
        for (std::size_t r = 0; r < R; ++r)
            f(m.at(r, c));
}

template <typename T, unsigned int R, unsigned int C, bool D>
T matrixSquareSum(const Matrix<T, R, C, D>& m) {
    T ret = 0;
    foreachElement(m, [&](const T& value) { ret += value * value; });
    return ret;
}

template <typename T, unsigned int S, bool Dyn>
struct LUDecomposition
{
    // PA = LU
    Matrix<T, S, S, Dyn> L;
    Matrix<T, S, S, Dyn> U;
    Matrix<T, S, S, Dyn> P;
};

template <typename T, unsigned int S, bool Dyn>
LUDecomposition<T, S, Dyn> LU(const Matrix<T, S, S, Dyn>& m) {
    LUDecomposition<T, S, Dyn> ret;
    ::LU(m.cref(), ret.P.ref(), ret.L.ref(), ret.U.ref());
    return ret;
}

template <typename T, unsigned int S, bool Dyn>
Matrix<T, S, S, Dyn> inverse(const Matrix<T, S, S, Dyn>& m) {
    Matrix<T, S, S, Dyn> ret;
    ::inverse(m.cref(), ret.ref());
    return ret;
}

template <typename T, typename V, unsigned int S, bool Dyn, bool Dyn2, bool Dyn3>
void solve(const Matrix<V, S, S, Dyn>& A, Matrix<T, S, 1, Dyn2>& x,
           const Matrix<T, S, 1, Dyn3>& b) {
    ::solve(A.cref(), x.ref(), b.cref());
}

template <typename T, typename V, unsigned int R, unsigned int C, bool Dyn, bool Dyn2, bool Dyn3>
void approximate(const Matrix<V, R, C, Dyn>& A, Matrix<T, C, 1, Dyn2>& x,
                 const Matrix<T, R, 1, Dyn3>& b) {
    ::approximate(A.cref(), x.ref(), b.cref());
}

template <typename T, unsigned int R, unsigned int C, bool D>
std::ostream& operator<<(std::ostream& out, const Matrix<T, R, C, D>& mat) {
    for (unsigned int r = 0; r < R; ++r) {
        out << "[";
        for (unsigned int c = 0; c < C; ++c) {
            out << mat.at(r, c);
            if (c + 1 < C)
                out << ", ";
        }
        out << "]\n";
    }
    return out;
}

template <typename T, unsigned int R, unsigned int C, bool D>
auto pseudoInverse(const Matrix<T, R, C, D>& mat) {
    return mat.transpose() * inverse(mat * mat.transpose());
}
