#pragma once

#include <algorithm>
#include <iostream>

#include <util.hpp>

#include "algorithms/findroot.hpp"
// #include "corecontext.h"
#include "memory.hpp"

struct RowMajorIndexing
{
    unsigned int operator()(unsigned int r, unsigned int c, unsigned int R, unsigned int C) const {
        ASSERT(r < R && c < C);
        return c + r * C;
    }
};

struct ColumnMajorIndexing
{
    unsigned int operator()(unsigned int r, unsigned int c, unsigned int R, unsigned int C) const {
        ASSERT(r < R && c < C);
        return r + c * R;
    }
};

template <typename I>
struct TransposeIndexing
{
    TransposeIndexing(const I& _indexer = {}) : indexer(_indexer) {}
    unsigned int operator()(unsigned int r, unsigned int c, unsigned int R, unsigned int C) const {
        return indexer(c, r, C, R);
    }

 private:
    I indexer;
};

template <typename I>
struct SubmatrixIndexing
{
    SubmatrixIndexing(unsigned int _r0, unsigned int _c0, unsigned int, unsigned int,
                      const SubmatrixIndexing& _indexer)
      : r0(_r0 + _indexer.r0),
        c0(_c0 + _indexer.c0),
        R(_indexer.R),
        C(_indexer.C),
        indexer(_indexer.indexer) {}
    SubmatrixIndexing(unsigned int _r0, unsigned int _c0, unsigned int _R, unsigned int _C,
                      const I& _indexer = {})
      : r0(_r0), c0(_c0), R(_R), C(_C), indexer(_indexer) {}
    unsigned int operator()(unsigned int r, unsigned int c, unsigned int, unsigned int) const {
        return indexer(r + r0, c + c0, R, C);
    }

    SubmatrixIndexing submatrix(unsigned int _r0, unsigned int _c0) const {
        return SubmatrixIndexing{r0 + _r0, c0 + _c0, R, C, indexer};
    }

 private:
    unsigned int r0;
    unsigned int c0;
    // original size
    unsigned int R;
    unsigned int C;
    I indexer;
};

template <typename T>
struct SubmatrixIndexingType
{ using type = SubmatrixIndexing<T>; };

template <typename I>
struct SubmatrixIndexingType<SubmatrixIndexing<I>>
{ using type = SubmatrixIndexing<I>; };

template <typename T>
using SubmatrixIndexing_t = typename SubmatrixIndexingType<T>::type;

template <typename T, typename I>
class ConstMatrixRef
{
 public:
    ConstMatrixRef(const T* _data, unsigned int _R, unsigned int _C, const I& _indexer = {})
      : data(_data), R(_R), C(_C), indexer(_indexer) {}

    const T& at(unsigned int r, unsigned int c) const { return data[indexer(r, c, R, C)]; }

    const T* getData() const { return data; }

    unsigned int columns() const { return C; }
    unsigned int rows() const { return R; }

    ConstMatrixRef<T, TransposeIndexing<I>> transpose() const {
        return ConstMatrixRef<T, TransposeIndexing<I>>(data, C, R, indexer);
    }

    ConstMatrixRef<T, SubmatrixIndexing_t<I>> submatrix(unsigned int r0, unsigned int c0,
                                                        unsigned int r, unsigned int c) const {
        ASSERT(r0 + r <= R && c0 + c <= C);
        return ConstMatrixRef<T, SubmatrixIndexing_t<I>>(
          data, r, c, SubmatrixIndexing_t<I>{r0, c0, R, C, indexer});
    }

    // template <typename I2>
    // ConstMatrixRef<T, I2> cref(const I2& ind = {}) const {
    //     return ConstMatrixRef<T, I2>(data, R, C, ind);
    // }

 private:
    const T* data;
    unsigned int R;
    unsigned int C;
    I indexer;
};

template <typename T, typename I>
class MatrixRef
{
 public:
    MatrixRef(T* _data, unsigned int _R, unsigned int _C, const I& _indexer = {})
      : data(_data), R(_R), C(_C), indexer(_indexer) {}

    T& at(unsigned int r, unsigned int c) const { return data[indexer(r, c, R, C)]; }

    T* getData() const { return data; }

    operator ConstMatrixRef<T, I>() const { return ConstMatrixRef(data, R, C, indexer); }

    // template <typename I2 = I>
    // MatrixRef<T, I2> ref(const I2& ind = {}) const {
    //     return MatrixRef<T, I2>(data, R, C, ind);
    // }

    // template <typename I2 = I>
    // ConstMatrixRef<T, I2> cref(const I2& ind = {}) const {
    //     return ConstMatrixRef<T, I2>(data, R, C, ind);
    // }

    ConstMatrixRef<T, I> cref() const { return ConstMatrixRef<T, I>(data, R, C, indexer); }

    unsigned int columns() const { return C; }
    unsigned int rows() const { return R; }

    MatrixRef<T, TransposeIndexing<I>> transpose() const {
        return MatrixRef<T, TransposeIndexing<I>>(data, C, R, indexer);
    }

    MatrixRef<T, SubmatrixIndexing_t<I>> submatrix(unsigned int r0, unsigned int c0, unsigned int r,
                                                   unsigned int c) const {
        ASSERT(r0 + r <= R && c0 + c <= C);
        return MatrixRef<T, SubmatrixIndexing_t<I>>(data, r, c,
                                                    SubmatrixIndexing_t<I>{r0, c0, R, C, indexer});
    }

    void swap(const MatrixRef& other) const {
        ASSERT(rows() == other.rows() && columns() == other.columns());
        using std::swap;
        for (unsigned int r = 0; r < rows(); ++r)
            for (unsigned int c = 0; c < columns(); ++c)
                swap(at(r, c), other.at(r, c));
    }
    friend void swap(const MatrixRef& a, const MatrixRef& b) { a.swap(b); }

 private:
    T* data;
    unsigned int R;
    unsigned int C;
    I indexer;
};

template <typename T>
class RowIterator
{
 public:
    using M = MatrixRef<T, RowMajorIndexing>;
    using value_type = M;
    using difference_type = int;
    using reference = const M&;
    RowIterator(unsigned int _row, const M& _mat) : row(_row), mat(_mat) {}

    void swap(RowIterator& other) {
        using std::swap;
        swap(row, other.row);
        for (unsigned int c = 0; c < mat.columns(); ++c)
            swap(mat.at(row, c), other.mat.at(row, c));
    }
    friend void swap(RowIterator& a, RowIterator& b) { a.swap(b); }

    RowIterator& operator++() {
        row++;
        return *this;
    }

    RowIterator operator++(int) {
        RowIterator ret = *this;
        row++;
        return ret;
    }

    RowIterator& operator--() {
        row--;
        return *this;
    }

    RowIterator operator--(int) {
        RowIterator ret = *this;
        row--;
        return ret;
    }

    RowIterator& operator+=(int n) {
        row += safe_cast<unsigned int>(n);
        return *this;
    }

    RowIterator& operator-=(int n) {
        row -= n;
        return *this;
    }

    friend RowIterator operator+(RowIterator itr, int n) {
        itr += n;
        return itr;
    }

    int operator-(const RowIterator& b) const {
        return static_cast<int>(row) - static_cast<int>(b.row);
    }

    friend RowIterator operator+(int n, RowIterator itr) {
        itr += n;
        return itr;
    }

    friend RowIterator operator-(int n, RowIterator itr) {
        itr -= n;
        return itr;
    }

    M operator[](unsigned int i) const { return M{&mat.at(row + i, 0), 1u, mat.columns()}; }
    M operator*() const { return (*this)[0]; }

    bool operator<(const RowIterator& other) const { return row < other.row; }
    bool operator>(const RowIterator& other) const { return row > other.row; }
    bool operator<=(const RowIterator& other) const { return row <= other.row; }
    bool operator>=(const RowIterator& other) const { return row >= other.row; }
    bool operator==(const RowIterator& other) const { return row == other.row; }
    bool operator!=(const RowIterator& other) const { return row != other.row; }

 private:
    unsigned int row;
    M mat;
};

namespace std
{
template <typename T>
struct iterator_traits<RowIterator<T>>
{
    using value_type = typename RowIterator<T>::value_type;
    using difference_type = typename RowIterator<T>::difference_type;
    using reference = typename RowIterator<T>::reference;
    // using pointer = typename RowIterator<T>::pointer;
    // using iterator_category = typename RowIterator<T>::pointer;
};
}  // namespace std

template <typename T, typename I>
void sage_matrix(std::ostream& out, const ConstMatrixRef<T, I>& mat) {
    out << "Matrix([";
    for (unsigned int r = 0; r < mat.rows(); ++r) {
        out << "[";
        for (unsigned int c = 0; c < mat.columns(); ++c) {
            out << static_cast<double>(mat.at(r, c));
            if (c + 1 < mat.columns())
                out << ", ";
        }
        out << "]";
        if (r + 1 < mat.rows())
            out << ", ";
    }
    out << "])";
}

template <typename T, typename I>
std::ostream& operator<<(std::ostream& out, const ConstMatrixRef<T, I>& mat) {
    ::sage_matrix(out, mat);
    return out;
}

template <typename T, typename I>
class TempMatrix
{
 public:
    explicit TempMatrix(unsigned int R, unsigned int C, StackMemory* memory, const I& _indexer = {})
      : data(R * C, memory), matRef(data.get(), R, C, _indexer) {}

    void init(const T& value) {
        for (unsigned int i = 0; i < matRef.rows() && i < matRef.columns(); ++i)
            matRef.at(i, i) = value;
    }

    void ident() {
        MEASURE("ident()");
        for (unsigned int r = 0; r < matRef.rows(); ++r)
            for (unsigned int c = 0; c < matRef.columns(); ++c)
                matRef.at(r, c) = r == c ? T{1} : T{0};
    }

    MatrixRef<T, I>&& ref() && { return std::move(matRef); }
    const MatrixRef<T, I>& ref() & { return matRef; }
    ConstMatrixRef<T, I> cref() const { return matRef.cref(); }

 private:
    StackMemory::MemoryHandle<T> data;
    MatrixRef<T, I> matRef;
};

template <typename T>
struct DynMatrixStorage_Dynamic
{
    std::vector<T> data;
    explicit DynMatrixStorage_Dynamic(unsigned int R, unsigned int C) : data(R * C) {}
    template <typename I>
    T& operator[](I i) {
        return data[i];
    }
    template <typename I>
    const T& operator[](I i) const {
        return data[i];
    }
    unsigned int size() const { return safe_cast<unsigned int>(data.size()); }
};

template <typename T>
struct DynMatrixStorage_Temp
{
    unsigned int _size = 0;
    StackMemory::MemoryHandle<T> handle;
    explicit DynMatrixStorage_Temp(unsigned int R, unsigned int C)
      : _size(R * C), handle(_size, TEMPMEM) {}
    template <typename I>
    T& operator[](I i) {
        return handle.get()[i];
    }
    template <typename I>
    const T& operator[](I i) const {
        return handle.get()[i];
    }
    unsigned int size() const { return _size; }
};

template <typename T, bool Temp>
struct DynMatrixStorageHelper;

template <typename T>
struct DynMatrixStorageHelper<T, true>
{ using type = DynMatrixStorage_Dynamic<T>; };

template <typename T>
struct DynMatrixStorageHelper<T, false>
{ using type = DynMatrixStorage_Temp<T>; };

template <typename T, bool Temp>
class DynMatrix
{
 public:
    using Indexing = RowMajorIndexing;
    explicit DynMatrix(unsigned int maxR, unsigned int _C) : C(_C), R(0), data(maxR, C) {}
    template <bool Temp2>
    explicit DynMatrix(unsigned int maxR, const DynMatrix<T, Temp2>& other)
      : C(other.columns()), R(other.rows()), data(maxR, C) {
        std::copy(std::begin(other.data), std::end(other.data), std::begin(data));
    }

    unsigned int columns() const { return C; }
    unsigned int rows() const { return R; }

    T* addRow() {
        ASSERT((R + 1) * C <= data.size());
        return &data[(R++) * C];
    }

    MatrixRef<T, RowMajorIndexing> ref() { return MatrixRef<T, RowMajorIndexing>(&data[0u], R, C); }

    ConstMatrixRef<T, RowMajorIndexing> cref() const {
        return ConstMatrixRef<T, RowMajorIndexing>(&data[0u], R, C);
    }

    // The data is not cleared, elements are not destructed
    void reset() { R = 0; }

 private:
    unsigned int C;
    unsigned int R;
    typename DynMatrixStorageHelper<T, Temp>::type data;
};

template <typename T, typename T2, typename I1, typename I2>
void copy(const ConstMatrixRef<T, I1>& a, const MatrixRef<T2, I2>& res) {
    MEASURE("copy(const mat&, mat&)");
    ASSERT(a.rows() == res.rows() && a.columns() == res.columns());
    for (unsigned int c = 0; c < a.columns(); ++c)
        for (unsigned int r = 0; r < a.rows(); ++r)
            res.at(r, c) = a.at(r, c);
}

template <typename T, typename T2, typename T3, typename I1, typename I2, typename I3>
void add(const ConstMatrixRef<T, I1>& a, const ConstMatrixRef<T2, I2>& b,
         const MatrixRef<T3, I3>& res) {
    ASSERT(a.rows() == b.rows() && a.columns() == b.columns());
    ASSERT(a.rows() == res.rows() && a.columns() == res.columns());
    for (unsigned int c = 0; c < a.columns(); ++c)
        for (unsigned int r = 0; r < a.rows(); ++r)
            res.at(r, c) = a.at(r, c) + b.at(r, c);
}

template <typename T, typename T2, typename I1, typename I2>
void add(const MatrixRef<T, I1>& a, const ConstMatrixRef<T2, I2>& b) {
    ASSERT(a.rows() == b.rows() && a.columns() == b.columns());
    for (unsigned int c = 0; c < a.columns(); ++c)
        for (unsigned int r = 0; r < a.rows(); ++r)
            a.at(r, c) += b.at(r, c);
}

template <typename T, typename T2, typename T3, typename I1, typename I2, typename I3>
void subs(const ConstMatrixRef<T, I1>& a, const ConstMatrixRef<T2, I2>& b, MatrixRef<T3, I3>& res) {
    ASSERT(a.rows() == b.rows() && a.columns() == b.columns());
    ASSERT(a.rows() == res.rows() && a.columns() == res.columns());
    for (unsigned int c = 0; c < a.columns(); ++c)
        for (unsigned int r = 0; r < a.rows(); ++r)
            res.at(r, c) = a.at(r, c) - b.at(r, c);
}

template <typename T, typename T2, typename I1, typename I2>
void subs(const MatrixRef<T, I1>& a, const ConstMatrixRef<T2, I2>& b) {
    ASSERT(a.rows() == b.rows() && a.columns() == b.columns());
    for (unsigned int c = 0; c < a.columns(); ++c)
        for (unsigned int r = 0; r < a.rows(); ++r)
            a.at(r, c) -= b.at(r, c);
}

template <typename T, typename T2, typename T3, typename I1, typename I2, typename I3>
void mul(const ConstMatrixRef<T, I1>& a, const ConstMatrixRef<T2, I2>& b,
         const MatrixRef<T3, I3>& res) {
    MEASURE("mul(const &)");
    INIT_ERROR(error(a, b));
    ASSERT(a.columns() == b.rows());
    ASSERT(a.rows() == res.rows() && b.columns() == res.columns());
    for (unsigned int c = 0; c < res.columns(); ++c) {
        for (unsigned int r = 0; r < res.rows(); ++r) {
            res.at(r, c) = T3{0};
            for (unsigned int k = 0; k < a.columns(); ++k)
                res.at(r, c) += a.at(r, k) * b.at(k, c);
        }
    }
    REGISTER_ERROR(error(res.cref()));
}

template <typename T, typename I, typename F>
void foreachElement(const MatrixRef<T, I>& m, F&& f) {
    for (unsigned int c = 0; c < m.columns(); ++c)
        for (unsigned int r = 0; r < m.rows(); ++r)
            f(m.at(r, c));
}

template <typename T, typename I, typename F>
void foreachElement(const ConstMatrixRef<T, I>& m, F&& f) {
    for (unsigned int c = 0; c < m.columns(); ++c)
        for (unsigned int r = 0; r < m.rows(); ++r)
            f(m.at(r, c));
}

template <typename T, typename I>
T matrixSquareSum(const ConstMatrixRef<T, I>& m) {
    T ret{0};
    foreachElement(m, [&](const T& value) { ret += value * value; });
    return ret;
}

// TODO
// template <typename T, typename I>
// T determinant(const ConstMatrixRef<T, I>& m) {
//     ASSERT(m.rows() == m.columns());
//     if (m.rows() == 1)
//         return m.at(0u, 0u);
//     else if (m.rows() == 2)

//         return m.at(0u, 0u) * m.at(1u, 1u) - m.at(1u, 0u) * m.at(0u, 1u);
//     else {
//         // T sum{0};
//         // for (unsigned int c = 0; c < m.rows(); ++c)
//         //     sum += m.at(0u, c) * determinant(removeRowAndColumn(m, static_cast<std::size_t>(0), c))
//         //            * static_cast<T>((c & 1) ? -1 : 1);
//         // return sum;
//     }
// }

template <typename T, typename I1, typename I2>
void inverse_lower(const ConstMatrixRef<T, I1>& m, const MatrixRef<T, I2>& inv) {
    MEASURE("inverse_lower(const &)");
    INIT_ERROR(error(m));
    ASSERT(m.rows() == m.columns());
    ASSERT(m.rows() == inv.rows() && m.columns() == inv.columns());
#ifdef DEBUG
    for (unsigned int c = 1; c < m.columns(); ++c)
        for (unsigned int r = 0; r < c; ++r)
            checkZero(m.at(r, c));
#endif
    for (unsigned int r = 0; r < m.rows(); ++r) {
        for (unsigned int c = 0; c <= r; ++c) {
            T sum = static_cast<T>(r == c ? 1 : 0);
            for (unsigned int k = 0; k < r; ++k)
                sum -= m.at(r, k) * inv.at(k, c);
            sum /= m.at(r, r);
            inv.at(r, c) = sum;
        }
        for (unsigned int c = r + 1; c < inv.columns(); ++c)
            inv.at(r, c) = T{0};
    }
    REGISTER_ERROR(error(inv.cref()));
}

template <typename T, typename I1, typename I2>
void inverse_upper(const ConstMatrixRef<T, I1>& m, const MatrixRef<T, I2>& inv) {
    MEASURE("inverse_upper(const &)");
    INIT_ERROR(error(m));
#ifdef DEBUG
    for (unsigned int c = 0; c < m.columns(); ++c)
        for (unsigned int r = c + 1; r < m.rows(); ++r)
            checkZero(m.at(r, c));
#endif
    // TODO if this is slow, implement the inverse here too
    inverse_lower(m.transpose(), inv.transpose());
    REGISTER_ERROR(error(inv.cref()));
}

template <typename T, typename I>
void gauss_elimination(const MatrixRef<T, I>& A) {
    MEASURE("gauss_elimination(&)");
    using std::abs;
    INIT_ERROR(error(A.cref()));
    ASSERT(A.rows() <= A.columns());
    for (unsigned int n = 0; n + 1 < A.rows(); ++n) {
        unsigned int m = n;
        for (unsigned int n2 = n + 1; n2 < A.rows(); ++n2) {
            if (abs(A.at(m, n)) < abs(A.at(n2, n)))
                m = n2;
        }
        if (m != n) {
            // Everything before n is 0 in these rows
            for (unsigned int c = n; c < A.columns(); ++c)
                std::swap(A.at(n, c), A.at(m, c));
        }
        const T Ann = A.at(n, n);
        if (Ann == T{0})
            continue;
        for (unsigned int r = A.rows() - 1; r > n; --r) {
            const T l = A.at(r, n) / Ann;
            A.at(r, n) = 0;
            for (unsigned int c = n + 1; c < A.columns(); ++c)
                A.at(r, c) -= l * A.at(n, c);
        }
    }
    REGISTER_ERROR(error(A.cref()));
}

template <typename T, typename I1, typename I2>
void gauss_elimination(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& res) {
    MEASURE("gauss_elimination(const&)");
    INIT_ERROR(error(A));
    ::copy(A, res);
    ::gauss_elimination(res);
    REGISTER_ERROR(error(res.cref()));
}

template <typename T, typename I>
void gauss_elimination_column(const MatrixRef<T, I>& A) {
    MEASURE("gauss_elimination_column(&)");
    using std::abs;
    INIT_ERROR(error(A.cref()));
    ASSERT(A.columns() <= A.rows());
    for (unsigned int n = 0; n + 1 < A.columns(); ++n) {
        unsigned int m = n;
        for (unsigned int n2 = n + 1; n2 < A.columns(); ++n2) {
            if (abs(A.at(n, m)) < abs(A.at(n, n2)))
                m = n2;
        }
        if (m != n) {
            // Everything before n is 0 in these columns
            for (unsigned int r = n; r < A.rows(); ++r)
                std::swap(A.at(r, n), A.at(r, m));
        }
        const T Ann = A.at(n, n);
        if (Ann == T{0})
            continue;
        for (unsigned int c = A.columns() - 1; c > n; --c) {
            const T l = A.at(n, c) / Ann;
            A.at(n, c) = 0;
            for (unsigned int r = n + 1; r < A.rows(); ++r)
                A.at(r, c) -= l * A.at(r, n);
        }
    }
    REGISTER_ERROR(error(A.cref()));
}

template <typename T, typename I1, typename I2>
void gauss_elimination_column(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& res) {
    MEASURE("gauss_elimination_column(const&)");
    INIT_ERROR(error(A));
    ::copy(A, res);
    ::gauss_elimination_column(res);
    REGISTER_ERROR(error(res.cref()));
}

template <typename T, bool Temp1, bool Temp2>
void kernel_right(DynMatrix<T, Temp1>& AI, DynMatrix<T, Temp2>& res, unsigned int forceCount = 0u) {
    // Ax = 0
    // https://en.wikipedia.org/wiki/Kernel_(linear_algebra)
    MEASURE("kernel_right(&)");
    INIT_ERROR(error(AI.cref()));
    ASSERT(res.rows() == 0);
    ASSERT(AI.columns() >= forceCount);
    unsigned int rows = AI.rows();
    for (unsigned int r = 0; r < AI.columns(); ++r) {
        T* row = AI.addRow();
        for (unsigned int c = 0; c < AI.columns(); ++c)
            row[c] = r == c ? T{1} : T{0};
    }
    auto ref = AI.ref();
    ::gauss_elimination_column(ref);
    for (unsigned int r = rows; r < AI.columns(); ++r) {
        T* row = res.addRow();
        for (unsigned int c = 0; c < AI.columns(); ++c)
            row[c] = ref.at(c + rows, r);
    }
    if (res.rows() < forceCount) {
        const unsigned int extra = forceCount - res.rows();
        struct RowData
        {
            unsigned int ind;
            T d;
        };
        StackMemory::MemoryHandle<RowData> permutationMem(AI.columns(), TEMPMEM);
        for (unsigned int c = 0; c < AI.columns(); ++c) {
            permutationMem.get()[c].ind = c;
            permutationMem.get()[c].d = T{0};
            for (unsigned int r = 0; r < rows; ++r)
                permutationMem.get()[c].d += ref.at(r, c) * ref.at(r, c);
        }
        std::nth_element(permutationMem.get(), permutationMem.get() + (extra - 1),
                         permutationMem.get() + (AI.columns() - extra),
                         [](const RowData& lhs, const RowData& rhs) { return lhs.d < rhs.d; });
        for (unsigned int i = 0; i < extra; ++i) {
            T* row = res.addRow();
            for (unsigned int c = 0; c < AI.columns(); ++c)
                row[c] = ref.at(c + rows, permutationMem.get()[i].ind);
        }
    }
    REGISTER_ERROR(error(res.cref()));
}

template <typename T, typename I, bool Temp>
void kernel_right(const ConstMatrixRef<T, I>& A, DynMatrix<T, Temp>& res,
                  unsigned int forceCount = 0u) {
    MEASURE("kernel_right(const&)");
    INIT_ERROR(error(A.cref()));
    DynMatrix<T, true> AI{A.columns() + A.rows(), A.columns()};
    for (unsigned int r = 0; r < A.rows(); ++r) {
        T* row = AI.addRow();
        for (unsigned int c = 0; c < A.columns(); ++c)
            row[c] = A.at(r, c);
    }
    ::kernel_right(AI, res, forceCount);
    REGISTER_ERROR(error(res.cref()));
}

template <typename T, typename I2, typename I3, typename I4>
void LU(const MatrixRef<T, I2>& P, const MatrixRef<T, I3>& L, const MatrixRef<T, I4>& A) {
    MEASURE("lu(&)");
    INIT_ERROR(error(A.cref()));
    // Doolittle algorithm from https://en.wikipedia.org/wiki/LU_decomposition
    ASSERT(A.rows() == A.columns());
    ASSERT(A.rows() == P.rows() && A.columns() == P.columns());
    ASSERT(A.rows() == L.rows() && A.columns() == L.columns());
#ifdef DEBUG
    for (unsigned int r = 0; r < A.rows(); ++r) {
        for (unsigned int c = 0; c < A.columns(); ++c) {
            if (r == c)
                ASSERT(P.at(r, c) == T{1} && L.at(r, c) == T{1});
            else
                ASSERT(P.at(r, c) == T{0} && L.at(r, c) == T{0});
        }
    }
#endif
    using std::abs;
    StackMemory::MemoryHandle<unsigned int> permutationMem(A.rows(), TEMPMEM);
    unsigned int* permutation = permutationMem.get();
    std::iota(permutation, permutation + A.rows(), 0);
    for (unsigned int n = 0; n + 1 < A.rows(); ++n) {
        unsigned int m = n;
        for (unsigned int n2 = n + 1; n2 < A.rows(); ++n2) {
            if (abs(A.at(m, n)) < abs(A.at(n2, n)))
                m = n2;
        }
        if (m != n) {
            std::swap(permutation[m], permutation[n]);
            // Everything before n is 0 in these rows
            for (unsigned int c = 0; c < n; ++c)
                std::swap(L.at(n, c), L.at(m, c));
            for (unsigned int c = n; c < A.columns(); ++c)
                std::swap(A.at(n, c), A.at(m, c));
        }
        const T Ann = A.at(n, n);
        for (unsigned int r = A.rows() - 1; r > n; --r) {
            const T l = A.at(r, n) / Ann;
            L.at(r, n) = l;
            A.at(r, n) = 0;
            for (unsigned int c = n + 1; c < A.columns(); ++c)
                A.at(r, c) -= l * A.at(n, c);
        }
    }
    for (unsigned int n = 0; n < A.rows(); ++n) {
        P.at(n, n) = 0;
        P.at(n, permutation[n]) = 1;
    }
    REGISTER_ERROR(error(P.cref(), L.cref(), A.cref()));
}

template <typename T, typename I1, typename I2, typename I3, typename I4>
void LU(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& P, const MatrixRef<T, I3>& L,
        const MatrixRef<T, I4>& U) {
    MEASURE("lu(const&)");
    INIT_ERROR(error(A));
    ::copy(A, U);
    ::LU(P, L, U);
    REGISTER_ERROR(error(P.cref(), L.cref(), U.cref()));
}

template <typename T, typename I, typename I2>
void inverse(const ConstMatrixRef<T, I>& m, const MatrixRef<T, I2>& inv) {
    MEASURE("inverse(const&)");
    INIT_ERROR(error(m));
    // m-1 = U-1*L-1*P
    TempMatrix<T, RowMajorIndexing> P{m.rows(), m.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> L{m.rows(), m.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> U{m.rows(), m.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> invLU{m.rows(), m.columns(), TEMPMEM};
    P.ident();
    L.ident();
    ::LU(m, P.ref(), L.ref(), U.ref());
    ::inverse_upper(U.cref(), inv);
    ::inverse_lower(L.cref(), U.ref());
    ::mul(inv.cref(), U.cref(), invLU.ref());  // inv=invU; U=invL
    ::mul(invLU.cref(), P.cref(), inv);
    REGISTER_ERROR(error(inv.cref()));
}

template <typename T, typename I, typename I2>
void inverse(MatrixRef<T, I>&& U, const MatrixRef<T, I2>& inv) {
    MEASURE("inverse(&&)");
    INIT_ERROR(error(U.cref()));
    // m-1 = U-1*L-1*P
#ifdef DEBUG
    TempMatrix<T, RowMajorIndexing> m{U.rows(), U.columns(), TEMPMEM};
    ::copy(U.cref(), m.ref());
#endif
    TempMatrix<T, RowMajorIndexing> P{U.rows(), U.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> L{U.rows(), U.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> invLU{U.rows(), U.columns(), TEMPMEM};
    P.ident();
    L.ident();
    ::LU(P.ref(), L.ref(), U);
    ::inverse_upper(U.cref(), inv);
    ::inverse_lower(L.cref(), U);
    ::mul(inv.cref(), U.cref(), invLU.ref());  // inv=invU; U=invL
    ::mul(invLU.cref(), P.cref(), inv);
    REGISTER_ERROR(error(inv.cref()));
}

template <typename T, typename V, typename I1, typename I2, typename I3>
void solve(const ConstMatrixRef<V, I1>& A, const MatrixRef<T, I2>& x,
           const ConstMatrixRef<T, I3>& b) {
    MEASURE("solve(const&)");
    INIT_ERROR(error(A, b));
    TempMatrix<V, RowMajorIndexing> inv{A.rows(), A.columns(), TEMPMEM};
    ::inverse(A, inv.ref());
    ::mul(inv.cref(), b, x);
    REGISTER_ERROR(error(x.cref()));
}

template <typename T, typename V, typename I1, typename I2, typename I3>
void solve(MatrixRef<V, I1>&& A, const MatrixRef<T, I2>& x, const ConstMatrixRef<T, I3>& b) {
    MEASURE("solve(&&)");
    INIT_ERROR(error(A.cref(), b));
    TempMatrix<V, RowMajorIndexing> inv{A.rows(), A.columns(), TEMPMEM};
    ::inverse(std::move(A), inv.ref());
    ::mul(inv.cref(), b, x);
    REGISTER_ERROR(error(x.cref()));
}

template <typename T, typename V, typename I1, typename I2, typename I3>
void approximate(const ConstMatrixRef<V, I1>& A, const MatrixRef<T, I2>& x,
                 const ConstMatrixRef<T, I3>& b) {
    MEASURE("approximate(const&)");
    INIT_ERROR(error(A, b));
    // argmin_x(Ax - b)
    // https://en.wikipedia.org/wiki/Overdetermined_system
    if (A.rows() == A.columns())
        ::solve(A, x, b);
    else {
        const auto At = A.transpose();
        TempMatrix<V, RowMajorIndexing> AtA{A.columns(), A.columns(), TEMPMEM};
        ::mul(At, A, AtA.ref());
        TempMatrix<V, RowMajorIndexing> inv{A.columns(), A.columns(), TEMPMEM};
        ::inverse(std::move(AtA).ref(), inv.ref());
        TempMatrix<V, RowMajorIndexing> m{A.columns(), A.rows(), TEMPMEM};
        ::mul(inv.cref(), At, m.ref());
        ::mul(m.cref(), b, x);
    }
    REGISTER_ERROR(error(x.cref()));
}

template <typename T, typename V, typename I1, typename I2, typename I3>
void approximate(MatrixRef<V, I1>&& A, const MatrixRef<T, I2>& x, const ConstMatrixRef<T, I3>& b) {
    MEASURE("approximate(&&)");
    INIT_ERROR(error(A.cref(), b));
    // argmin_x(Ax - b)
    // https://en.wikipedia.org/wiki/Overdetermined_system
    if (A.rows() == A.columns())
        ::solve(std::move(A), x, b);
    else {
        const auto At = A.transpose();
        TempMatrix<V, RowMajorIndexing> AtA{A.columns(), A.columns(), TEMPMEM};
        ::mul(At.cref(), A.cref(), AtA.ref());
        TempMatrix<V, RowMajorIndexing> inv{A.columns(), A.columns(), TEMPMEM};
        ::inverse(std::move(AtA).ref(), inv.ref());
        TempMatrix<V, RowMajorIndexing> m{A.columns(), A.rows(), TEMPMEM};
        ::mul(inv.cref(), At.cref(), m.ref());
        ::mul(m.cref(), b, x);
    }
    REGISTER_ERROR(error(x.cref()));
}

template <typename T, typename I1, typename I2>
T dot(const ConstMatrixRef<T, I1>& a, const ConstMatrixRef<T, I2>& b) {
    ASSERT(a.rows() == b.rows() && a.columns() == 1u && b.columns() == 1u);
    T ret = T{0};
    for (unsigned int r = 0; r < a.rows(); ++r)
        ret += a.at(r, 0u) * b.at(r, 0u);
    return ret;
}

template <typename T, typename I>
T lengthSq(const ConstMatrixRef<T, I>& m) {
    return ::dot(m, m);
}

template <typename T, typename I>
T length(const ConstMatrixRef<T, I>& m) {
    using math::sqrt;
    return sqrt(::lengthSq(m));
}

template <typename T, typename I>
void normalize(const MatrixRef<T, I>& m) {
    ASSERT(m.columns() == 1u);
    const T len = ::length(m.cref());
    for (unsigned int r = 0; r < m.rows(); ++r)
        m.at(r, 0u) /= len;
}

template <typename T, typename I1, typename I2>
void house_holder(const ConstMatrixRef<T, I1>& vec, const MatrixRef<T, I2>& res) {
    MEASURE("house_holder(&)");
#ifdef DEBUG
    checkZero(::dot(vec, vec) - T{1});
#endif
    INIT_ERROR(error(vec));
    ASSERT(res.rows() == vec.rows() && res.columns() == vec.rows());
    ASSERT(vec.columns() == 1u);
    for (unsigned int r = 0; r < res.rows(); ++r)
        for (unsigned int c = 0; c < res.columns(); ++c)
            res.at(r, c) = (r == c ? T{1} : T{0}) - T{2} * vec.at(r, 0u) * vec.at(c, 0u);
    REGISTER_ERROR(error(res.cref()));
}

//! computes AH
//! A and res can be the same
template <typename T, typename I1, typename I2, typename I3>
void mul_house_holder(const ConstMatrixRef<T, I2>& A, const MatrixRef<T, I3>& res,
                      const ConstMatrixRef<T, I1>& vec) {
    // AH = A - 2/||v|| * (Av)v^T
    MEASURE("mul_house_holder(const &A, H)");
    INIT_ERROR(error(A, vec));
    ASSERT(res.rows() == A.rows() && res.columns() == A.columns());
    ASSERT(vec.rows() == A.columns());
    ASSERT(vec.columns() == 1u);
    TempMatrix<T, RowMajorIndexing> v{A.rows(), 1u, TEMPMEM};
    auto vRef = v.cref();
    ::mul(A, vec, v.ref());
    const T m = T{2} / ::dot(vec, vec);
    for (unsigned int r = 0; r < res.rows(); ++r)
        for (unsigned int c = 0; c < res.columns(); ++c)
            res.at(r, c) = A.at(r, c) - vRef.at(r, 0u) * vec.at(c, 0u) * m;
    REGISTER_ERROR(error(res.cref()));
}

//! computes AH
template <typename T, typename I1, typename I2>
void mul_house_holder(const MatrixRef<T, I2>& A, const ConstMatrixRef<T, I1>& vec) {
    MEASURE("mul_house_holder(&A, H)");
    INIT_ERROR(error(A.cref(), vec));
    ::mul_house_holder(A.cref(), A, vec);
    REGISTER_ERROR(error(A.cref()));
}

//! computes HA
//! A and res can be the same
template <typename T, typename I1, typename I2, typename I3>
void mul_house_holder(const ConstMatrixRef<T, I1>& vec, const ConstMatrixRef<T, I2>& A,
                      const MatrixRef<T, I3>& res) {
    MEASURE("mul_house_holder(const &A, H)");
    INIT_ERROR(error(vec, A));
    ASSERT(res.rows() == A.rows() && res.columns() == A.columns());
    ASSERT(vec.rows() == A.rows());
    ASSERT(vec.columns() == 1u);

    // AH = A - 2/||v|| * (Av)v^T
    // HA = (A^T H^T)^T
    // HA = (A^T H)^T
    // A^T H = A^T - 2/||v|| * (A^Tv)v^T

    TempMatrix<T, RowMajorIndexing> v{A.columns(), 1u, TEMPMEM};
    auto vRef = v.cref();
    ::mul(A.transpose(), vec, v.ref());
    const T m = T{2} / ::dot(vec, vec);
    for (unsigned int r = 0; r < res.rows(); ++r)
        for (unsigned int c = 0; c < res.columns(); ++c)
            res.at(r, c) = A.at(r, c) - vRef.at(c, 0u) * vec.at(r, 0u) * m;
    REGISTER_ERROR(error(res.cref()));
}

//! computes HA
template <typename T, typename I1, typename I2>
void mul_house_holder(const ConstMatrixRef<T, I1>& vec, const MatrixRef<T, I2>& A) {
    MEASURE("mul_house_holder(H, &A)");
    MEASURE("mul_house_holder(const &A, H)");
    ::mul_house_holder(vec, A.cref(), A);
    REGISTER_ERROR(error(A.cref()));
}

template <typename T, typename I>
void givens_matrix(const MatrixRef<T, I>& mat, unsigned int j, unsigned int i, const T& s,
                   const T& c) {
#ifdef DEBUG
    for (unsigned int r = 0; r < mat.rows(); ++r) {
        for (unsigned int cc = 0; cc < mat.columns(); ++cc) {
            if (r == cc)
                ASSERT(mat.at(r, cc) == T{1});
            else
                ASSERT(mat.at(r, cc) == T{0});
        }
    }
#endif
    mat.at(j, j) = c;
    mat.at(i, i) = c;
    mat.at(i, j) = s;
    mat.at(j, i) = -s;
}

template <typename T, typename I>
void apply_givens(const MatrixRef<T, I>& mat, unsigned int j, unsigned int i, const T& s,
                  const T& c) {
    for (unsigned int r = 0; r < mat.rows(); ++r) {
        const auto mj = mat.at(r, j);
        const auto mi = mat.at(r, i);
        mat.at(r, j) = c * mj + s * mi;
        mat.at(r, i) = -s * mj + c * mi;
    }
}

template <typename T, typename I>
void apply_givens(const MatrixRef<T, I>& mat, unsigned int j, unsigned int i, const T& angle) {
    using math::cos;
    using math::sin;
    ::apply_givens(mat, i, j, sin(angle), cos(angle));
}

template <typename T, typename I>
void apply_givens(unsigned int j, unsigned int i, const T& s, const T& c,
                  const MatrixRef<T, I>& mat) {
    for (unsigned int cc = 0; cc < mat.columns(); ++cc) {
        const auto mj = mat.at(j, cc);
        const auto mi = mat.at(i, cc);
        mat.at(j, cc) = c * mj - s * mi;
        mat.at(i, cc) = s * mj + c * mi;
    }
}

template <typename T, typename I>
void apply_givens(unsigned int j, unsigned int i, const T& angle, const MatrixRef<T, I>& mat) {
    using math::cos;
    using math::sin;
    ::apply_givens(i, j, sin(angle), cos(angle), mat);
}

//! Transforms (x,y) to (r, 0)
template <typename T>
void get_givens_params(const T& x, const T& y, T& s, T& c) {
    using math::hypot;
    T r = hypot(x, y);
    T invR = T{1} / r;
    c = x * invR;
    s = -y * invR;
    ASSERT(c == c && s == s);
}

template <typename T, typename I1, typename I2>
void QR(const MatrixRef<T, I1>& Q, const MatrixRef<T, I2>& R) {
    MEASURE("qr(&)");
    INIT_ERROR(error(R.cref()));
    ASSERT(R.rows() >= R.columns());
    ASSERT(Q.rows() == R.rows() && Q.columns() == R.rows());
#ifdef DEBUG
    for (unsigned int r = 0; r < Q.rows(); ++r) {
        for (unsigned int c = 0; c < Q.columns(); ++c) {
            if (r == c)
                ASSERT(Q.at(r, c) == T{1});
            else
                ASSERT(Q.at(r, c) == T{0});
        }
    }
#endif
    using std::abs;
    for (unsigned int c = 0; c < R.columns(); ++c) {
        for (unsigned int r = R.rows() - 1; r > c; --r) {
            T b = R.at(r, c);
            if (abs(b) > T{0}) {
                unsigned int ind = r - 1;
                T a = R.at(ind, c);
                T sn, cs;
                ::get_givens_params(a, b, sn, cs);
                ::apply_givens(ind, r, sn, cs, R);
                R.at(r, c) = T{0};                   // avoid some computational error
                ::apply_givens(Q, ind, r, -sn, cs);  // transpose
            }
        }
    }
    REGISTER_ERROR(error(Q.cref(), R.cref()));
}

template <typename T, typename I1, typename I2, typename I3>
void QR(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& Q, const MatrixRef<T, I3>& R) {
    MEASURE("qr(const&)");
    INIT_ERROR(error(A.cref()));
    ::copy(A, R);
    ::QR(Q, R);
    REGISTER_ERROR(error(Q.cref(), R.cref()));
}

template <typename T, typename I>
void normalize_columns(const MatrixRef<T, I>& m) {
    MEASURE("normalize_columns(&)");
    INIT_ERROR(error(m.cref()));
    for (unsigned int c = 0; c < m.columns(); ++c) {
        T val{0};
        for (unsigned int r = 0; r < m.rows(); ++r)
            val += m.at(r, c) * m.at(r, c);
        using math::sqrt;
        if (val > 0) {
            val = T{1} / sqrt(val);
            for (unsigned int r = 0; r < m.rows(); ++r)
                m.at(r, c) *= val;
        }
    }
    REGISTER_ERROR(error(m.cref()));
}

// template <typename T, typename I1, typename I2, typename I3>
// void find_eigen_vectors(const ConstMatrixRef<T, I1>& A, const ConstMatrixRef<T, I2>& D,
//                         const MatrixRef<T, I3>& Q) {
//     // solve (A - ai*I) * vij = 0
//     MEASURE("find_eigen_vectors(const&)");
//     ASSERT(A.rows() == A.columns());
//     ASSERT(Q.rows() == A.rows() && Q.columns() == A.columns());
//     ASSERT(D.rows() == A.rows() && D.columns() == A.columns());
// #ifdef DEBUG
//     using std::abs;
//     for (unsigned int r = 1; r < D.rows(); ++r)
//         ASSERT(abs(D.at(r + 1, r + 1)) < abs(D.at(r, r)));
// #endif
//         // (A - ai*I) * vij = 0
//         // P*(A - ai*I) = LDU
//         // (A - ai*I) = LDU * P-1
//         // LDU * P-1 * vij = 0
//         // DU * P-1 * vij = L-1 * 0
// #ifdef DEBUG
//     for (unsigned int i = 0; i < D.rows(); ++i) {
//         for (unsigned int j = 0; j < D.rows(); ++j) {
//             T sum = T{0};
//             for (unsigned int k = 0; k < D.rows(); ++k)
//                 sum += Q.at(k, i) * A.at(j, k);
//             checkZero(sum - D.at(i, i) * Q.at(j, i));
//         }
//     }
// #endif
// }

// template <typename T, typename I1, typename I2, typename I3>
// void eigen_decomposition(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& Q,
//                          const MatrixRef<T, I3>& D) {
//     // http://hua-zhou.github.io/teaching/biostatm280-2017spring/slides/16-eigsvd/eigsvd.html
//     // A = QDQ-1
//     MEASURE("eigen_decomposition(const&)");
//     ASSERT(A.rows() == A.columns());
//     ASSERT(Q.rows() == A.rows() && Q.columns() == A.columns());
//     ASSERT(D.rows() == A.rows() && D.columns() == A.columns());
// #ifdef DEBUG
//     for (unsigned int r = 0; r < Q.rows(); ++r) {
//         for (unsigned int c = 0; c < Q.columns(); ++c) {
//             if (r != c)
//                 ASSERT(D.at(r, c) == T{0});
//         }
//     }
// #endif
//     if (A.rows() > 3) {
//         // TODO implement this
//         ASSERT(false);
//     }
//     else {
//         // Analytic solution as available
//         if (A.rows() == 3) {
//             // -((a - a11)*(a - a22) - a12*a21)*(a - a00) + ((a - a22)*a10 + a12*a20)*a01 + ((a - a11)*a20 + a10*a21)*a02 = 0
//             // -a^3 + a^2*a00 + a*a01*a10 + a^2*a11 - a*a00*a11 + a*a02*a20 - a02*a11*a20 + a01*a12*a20 + a02*a10*a21 + a*a12*a21 - a00*a12*a21 + a^2*a22 - a*a00*a22 - a01*a10*a22 - a*a11*a22 + a00*a11*a22
//             // a^3 * (-1)
//             // a^2 * (a00 + a11 + a22)
//             // a * (a01*a10 - a00*a11 + a02*a20 + a12*a21 - a00*a22 - a11*a22)
//             //  - a02*a11*a20 + a01*a12*a20 + a02*a10*a21 - a00*a12*a21 - a01*a10*a22 + a00*a11*a22
//             const auto& a00 = A.at(0u, 0u);
//             const auto& a01 = A.at(0u, 1u);
//             const auto& a02 = A.at(0u, 2u);
//             const auto& a10 = A.at(1u, 0u);
//             const auto& a11 = A.at(1u, 1u);
//             const auto& a12 = A.at(1u, 2u);
//             const auto& a20 = A.at(2u, 0u);
//             const auto& a21 = A.at(2u, 1u);
//             const auto& a22 = A.at(2u, 2u);
//             ::find_roots(T{-1}, a00 + a11 + a22,
//                          a01 * a10 - a00 * a11 + a02 * a20 + a12 * a21 - a00 * a22 - a11 * a22,
//                          -a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
//                            - a01 * a10 * a22 + a00 * a11 * a22,
//                          D.at(0u, 0u), D.at(1u, 1u), D.at(2u, 2u));
//         }
//         else {
//             ASSERT(A.rows() == 2);
//             // (b - b00)*(b - b11) - b01*b10 = 0
//             // (b - b00)*(b - b11) = b01*b10
//             // b^2 + b*( - b00 - b11) + (b00*b11 - b01*b10) = 0
//             ::find_roots(T{1}, -A.at(0u, 0u) - A.at(1u, 1u),
//                          A.at(0u, 0u) * A.at(1u, 1u) - A.at(0u, 1u) * A.at(1u, 0u), D.at(0u, 0u),
//                          D.at(1u, 1u));
//         }
//         using std::abs;
//         using std::swap;
//         for (unsigned int i = 0; i + 1 < A.rows(); ++i)
//             for (unsigned int j = i + 1; j < A.rows(); ++j)
//                 if (abs(D.at(i, i)) < abs(D.at(j, j)))
//                     swap(D.at(i, i), D.at(j, j));
//         ::find_eigen_vectors(A, D.cref(), Q);
//     }
//     ::normalize_columns(Q);

// #ifdef DEBUG
//     TempMatrix<T, RowMajorIndexing> m{A.rows(), A.columns(), TEMPMEM};
//     TempMatrix<T, RowMajorIndexing> m2{A.rows(), A.columns(), TEMPMEM};
//     ::mul(Q.cref(), D.cref(), m.ref());
//     ::mul(m.cref(), Q.cref().transpose(), m2.ref());
//     ::subs(m2.ref(), A.cref());
//     auto s = ::matrixSquareSum(m2.cref());
//     checkZer(s);
// #endif
// }

// template <typename T, typename I1, typename I2, typename I3, typename I4>
// void SVD(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& U, const MatrixRef<T, I3>& S,
//          const MatrixRef<T, I4>& Vt) {
//     // http://www.rohitab.com/discuss/topic/36251-c-svd-of-3x3-matrix/
//     MEASURE("svd(const&)");
//     ASSERT(U.rows() == A.rows() && U.columns() == A.rows());
//     ASSERT(Vt.rows() == A.columns() && Vt.columns() == A.columns());
//     ASSERT(S.rows() == A.rows() && S.columns() == A.columns());
//     // A = USVt
//     // AtA = (VStUt) * (USVt) = V * (St*S) * Vt
//     // AAt = (USVt) * (VStUt) = U * (S*St) * Ut
//     const auto At = A.transpose();
//     TempMatrix<T, RowMajorIndexing> AtA{A.columns(), A.columns(), TEMPMEM};
//     TempMatrix<T, RowMajorIndexing> AAt{A.rows(), A.rows(), TEMPMEM};
//     TempMatrix<T, RowMajorIndexing> D{A.columns(), A.columns(), TEMPMEM};
//     TempMatrix<T, RowMajorIndexing> D2{A.rows(), A.rows(), TEMPMEM};
//     ::mul(At, A, AtA.ref());
//     ::mul(A, At, AAt.ref());
//     // AtA = V * D * Vt = V * (St * S) * Vt
//     ::eigen_decomposition(AtA.cref(), Vt.transpose(), D.ref());
//     // TODO this should be avoidable
//     // AAt = U * D2 * Ut = U * (S*St) * Ut
//     ::eigen_decomposition(AAt.cref(), U, D2.ref());
//     // rank(S) = 0: U = V
//     // A*V * S-1 = U (if S-1 exists)

//     using math::sqrt;
//     for (unsigned int i = 0; i < std::min(S.rows(), S.columns()); ++i)
//         S.at(i, i) = sqrt((D.cref().at(i, i) + D2.cref().at(i, i)) * Float{0.5});

// #ifdef DEBUG
//     TempMatrix<T, RowMajorIndexing> m{A.rows(), A.columns(), TEMPMEM};
//     TempMatrix<T, RowMajorIndexing> m2{A.rows(), A.columns(), TEMPMEM};
//     ::mul(U.cref(), S.cref(), m.ref());
//     ::mul(m.cref(), Vt.cref(), m2.ref());
//     ::subs(m2.ref(), A.cref());
//     auto s = ::matrixSquareSum(m2.cref());
//     checkZer(s);
// #endif
// }

//! B = QBP^t
template <typename T, typename I2, typename I3, typename I4>
void bidiagonalize(const MatrixRef<T, I2>& Q, const MatrixRef<T, I3>& A,
                   const MatrixRef<T, I4>& Pt) {
    // A possible improvement: https://epubs.siam.org/doi/10.1137/S0895479898343541
    using math::sqrt;
    MEASURE("bidiagonalize(&)");
    INIT_ERROR(error(A.cref()));
#ifdef DEBUG
    for (unsigned int r = 0; r < Q.rows(); ++r) {
        for (unsigned int c = 0; c < Q.columns(); ++c) {
            if (r == c)
                ASSERT(Q.at(r, c) == T{1});
            else
                ASSERT(Q.at(r, c) == T{0});
        }
    }
    for (unsigned int r = 0; r < Pt.rows(); ++r) {
        for (unsigned int c = 0; c < Pt.columns(); ++c) {
            if (r == c)
                ASSERT(Pt.at(r, c) == T{1});
            else
                ASSERT(Pt.at(r, c) == T{0});
        }
    }
#endif
    ASSERT(A.rows() >= A.columns());
    ASSERT(Q.rows() == A.rows() && Q.columns() == A.rows());
    ASSERT(Pt.rows() == A.columns() && Pt.columns() == A.columns());
    TempMatrix<T, RowMajorIndexing> v{std::max(A.rows(), A.columns()), 1u, TEMPMEM};

    for (unsigned int i = 0; i < std::min(A.rows(), A.columns()); ++i) {
        // A' = H1AG1
        // A'' = H2H1AG1G2
        // H1H2A''G2G1 = A
        auto Aj_plus = A.submatrix(i, i + 1, A.rows() - i, A.columns() - i - 1);
        auto vj = v.ref().submatrix(0u, 0u, A.rows() - i, 1u);
        {  // column
            auto Aj = A.submatrix(i, i, A.rows() - i, A.columns() - i);
            T b{0};
            for (unsigned int r = 0; r < Aj.rows(); ++r) {
                vj.at(r, 0u) = Aj.at(r, 0u);
                b += Aj.at(r, 0u) * Aj.at(r, 0u);
            }
            if (b > Aj.at(0u, 0u) * Aj.at(0u, 0u)) {
                b = sqrt(b);
                vj.at(0u, 0u) -= b;  // v = (a1 - b*e1)
                ::normalize(vj);
                ::mul_house_holder(vj.cref(), Aj_plus);
                ::mul_house_holder(Q.submatrix(0u, i, Q.rows(), Q.columns() - i), vj.cref());
                Aj.at(0u, 0u) = b;
                for (unsigned int r = 1; r < Aj.rows(); ++r)
                    Aj.at(r, 0u) = T{0};
            }
        }
        if (A.columns() - i - 1 > 1) {  // row
            auto vj_plus = v.ref().submatrix(0u, 0u, A.columns() - i - 1, 1u);
            auto Aj_next = A.submatrix(i + 1, i + 1, A.rows() - i - 1, A.columns() - i - 1);
            T a{0};
            for (unsigned int c = 0; c < Aj_plus.columns(); ++c) {
                vj_plus.at(c, 0u) = Aj_plus.at(0u, c);
                a += Aj_plus.at(0u, c) * Aj_plus.at(0u, c);
            }
            if (a > Aj_plus.at(0u, 0u) * Aj_plus.at(0u, 0u)) {
                a = sqrt(a);
                vj_plus.at(0u, 0u) -= a;
                ::normalize(vj_plus);
                ::mul_house_holder(Aj_next, vj_plus.cref());
                ::mul_house_holder(vj_plus.cref(),
                                   Pt.submatrix(i + 1, 0u, Pt.rows() - i - 1, Pt.columns()));
                Aj_plus.at(0u, 0u) = a;
                for (unsigned int c = 1; c < Aj_plus.columns(); ++c)
                    Aj_plus.at(0u, c) = T{0};
            }
        }
    }
    REGISTER_ERROR(error(Q.cref(), A.cref(), Pt.cref()));
}

//! A = QBP^t
template <typename T, typename I1, typename I2, typename I3, typename I4>
void bidiagonalize(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& Q,
                   const MatrixRef<T, I3>& B, const MatrixRef<T, I4>& Pt) {
    MEASURE("bidiagonalize(const&)");
    INIT_ERROR(error(A));
    ASSERT(B.rows() == A.rows() && B.columns() == A.columns());
    ::copy(A, B);
    ::bidiagonalize(Q, B, Pt);
    REGISTER_ERROR(error(Q.cref(), B.cref(), Pt.cref()));
}

//! Wilkinson shift for [[a, b], [b, c]]
template <typename T>
T wilkinson_shift(const T& a, const T& b, const T& c, const T& target) {
    using std::abs;
    T x1, x2;
    ::find_roots_forced(T{1}, -(a + c), a * c - b * b, x1, x2);
    T ret = abs(x1 - target) < abs(x2 - target) ? x1 : x2;
    ASSERT(ret == ret);
    return ret;
}

template <typename T>
struct IterationCondition
{
    IterationCondition(const T& _maxError) : maxError(_maxError) {}
    IterationCondition(unsigned int _maxSteps, const T& _maxError)
      : maxSteps(_maxSteps), maxError(_maxError) {}

    bool operator()(unsigned int steps, const T& error) const {
        return steps < maxSteps && error > maxError;
    }

    bool isZero(const T& error) const {
        using std::abs;
        return abs(error) <= maxError;
    }

    bool operator()(const T& error) const { return error > maxError; }

    unsigned int maxSteps = std::numeric_limits<unsigned int>::max();
    T maxError;
};

//! B = X B' Y^T (B is bidiagonal)
template <typename T, typename I1, typename I2, typename I3, typename F = IterationCondition<T>>
void diagonalize(const MatrixRef<T, I2>& X, const MatrixRef<T, I1>& B, const MatrixRef<T, I3>& Yt,
                 unsigned int n0, unsigned int rows, unsigned int columns, F&& condition) {
    // http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/QRalgorithm.html
    MEASURE("diagonalize(&, rec)");
    INIT_ERROR(error(X.cref(), B.cref(), Yt.cref()));
    auto subB = B.submatrix(n0, n0, rows, columns);
    for (unsigned int i = std::min(rows, columns); i > 1; i--) {
        using std::abs;
        unsigned int step = 0;
        while (condition(++step, abs(subB.at(i - 2, i - 1)))) {
            MEASUREMENT_INC(ITERATION);
            for (unsigned int k = 0; k + 1 < i; ++k) {
                if (condition.isZero(subB.at(k, k + 1))) {
                    subB.at(k, k + 1) = T{0};
                    if (k + 1 > 1)
                        ::diagonalize(X, B, Yt, n0, k + 1, k + 1, condition);
                    if (i - k - 1 > 1)
                        ::diagonalize(X, B, Yt, n0 + k + 1, i - k - 1, i - k - 1, condition);
                    return;
                }
                if (condition.isZero(subB.at(k, k))) {
                    subB.at(k, k) = T{0};
                    // this value would be smeared over the entire column
                    // insignificant, can be discarded
                    for (unsigned int c = k + 1; c < subB.columns(); ++c) {
                        if (condition.isZero(subB.at(k, c))) {
                            subB.at(k, c) = T{0};
                            break;
                        }
                        T sn, cs;
                        ::get_givens_params(subB.at(k, c), subB.at(c, c), sn, cs);
                        // switch rows
                        std::swap(sn, cs);
                        cs = -cs;
                        // This could be improved
                        ::apply_givens(k, c, sn, cs, subB);
                        subB.at(k, c) = T{0};  // avoid some computational error
                        ::apply_givens(X, k + n0, c + n0, -sn, cs);  // transpose
                    }
                    if (k + 1 > 1)
                        ::diagonalize(X, B, Yt, n0, k + 1, k + 1, condition);
                    if (i - k - 1)
                        ::diagonalize(X, B, Yt, n0 + k + 1, i - k - 1, i - k - 1, condition);
                    return;
                }
            }

            const T qk_0 = subB.at(i - 2, i - 2);
            const T qk_1 = subB.at(i - 1, i - 1);
            const T ek_0 = i >= 3 ? subB.at(i - 3, i - 2) : T{0};
            const T ek_1 = subB.at(i - 2, i - 1);
            // etries of B^t*B's lower right 2x2 submatrix
            const T s = ::wilkinson_shift(qk_0 * qk_0 + ek_0 * ek_0, ek_1 * qk_0,
                                          ek_1 * ek_1 + qk_1 * qk_1, qk_1);
            // Initial rotation
            T sn, cs;
            // B^T * B's first column
            const auto& d1 = subB.at(0u, 0u);
            const auto& e2 = subB.at(0u, 1u);
            ::get_givens_params(d1 * d1 - s, e2 * d1, sn, cs);
            sn = -sn;
            ::apply_givens(subB.submatrix(0u, 0u, 2u, 2u), 0u, 1u, sn, cs);
            ::apply_givens(n0, 1u + n0, -sn, cs, Yt);  // transpose
            for (unsigned int k = 0; k + 1 < i; ++k) {
                if (condition.isZero(subB.at(k + 1, k))) {
                    subB.at(k + 1, k) = T{0};
                    break;
                }
                // B' = P2*P1*B*G1*G2
                // P1^T*P2^T * B' * G2^T*G1^T = B
                ::get_givens_params(subB.at(k, k), subB.at(k + 1, k), sn, cs);
                ::apply_givens(0u, 1u, sn, cs, subB.submatrix(k, k, 2u, std::min(3u, i - k)));
                subB.at(k + 1, k) = T{0};                        // avoid some computational error
                ::apply_givens(X, n0 + k, n0 + k + 1, -sn, cs);  // transpose
                if (k + 2 < i) {
                    // new entry has been created to Bi.at(k, k+2)
                    if (condition.isZero(subB.at(k, k + 2))) {
                        subB.at(k, k + 2) = T{0};
                        break;
                    }
                    ::get_givens_params(subB.at(k, k + 1), subB.at(k, k + 2), sn, cs);
                    ::apply_givens(subB.submatrix(k, k, std::min(i - k, 3u), std::min(i - k, 3u)),
                                   1u, 2u, -sn, cs);
                    subB.at(k, k + 2) = T{0};  // avoid some computational error
                    ::apply_givens(n0 + k + 1, n0 + k + 2, sn, cs, Yt);  // transpose
                }
            }
        }
        subB.at(i - 2, i - 1) = T{0};
    }
    REGISTER_ERROR(error(X.cref(), B.cref(), Yt.cref()));
}

template <typename T, typename I1, typename I2, typename I3, typename F = IterationCondition<T>>
void diagonalize(const MatrixRef<T, I2>& X, const MatrixRef<T, I1>& B, const MatrixRef<T, I3>& Yt,
                 F&& condition) {
    MEASURE("diagonalize(&)");
    INIT_ERROR(error(B.cref()));
    ::diagonalize(X, B, Yt, 0u, B.rows(), B.columns(), std::forward<F>(condition));
    REGISTER_ERROR(error(X.cref(), B.cref(), Yt.cref()));
}

//! B = XSY^T (B is bidiagonal)
template <typename T, typename I1, typename I2, typename I3, typename I4,
          typename F = IterationCondition<T>>
void diagonalize(const ConstMatrixRef<T, I1>& B, const MatrixRef<T, I2>& X,
                 const MatrixRef<T, I3>& S, const MatrixRef<T, I4>& Yt, F&& condition) {
    MEASURE("diagonalize(const&)");
    INIT_ERROR(error(B.cref()));
    ASSERT(S.rows() == B.rows() && S.columns() == B.columns());
    ::copy(B, S);
    ::diagonalize(X, S, Yt, 0u, B.rows(), B.columns(), std::forward<F>(condition));
    REGISTER_ERROR(error(X.cref(), S.cref(), Yt.cref()));
}

//! Golub-Reinsch algorithm
template <typename T, typename I1, typename I2, typename I3, typename I4,
          typename F = IterationCondition<T>>
void SVD(const ConstMatrixRef<T, I1>& A, const MatrixRef<T, I2>& U, const MatrixRef<T, I3>& S,
         const MatrixRef<T, I4>& Vt, F&& condition) {
    // http://www.rohitab.com/discuss/topic/36251-c-svd-of-3x3-matrix/
    // https://scicomp.stackexchange.com/questions/6979/how-is-the-svd-of-a-matrix-computed-in-practice
    // This can be improved for rows >> columns with 'chan.pdf'
    MEASURE("svd(const&)");
    INIT_ERROR(error(A));
    ASSERT(A.rows() >= A.columns());
    ASSERT(U.rows() == A.rows() && U.columns() == A.rows());
    ASSERT(Vt.rows() == A.columns() && Vt.columns() == A.columns());
    ASSERT(S.rows() == A.rows() && S.columns() == A.columns());
#ifdef DEBUG
    for (unsigned int r = 0; r < U.rows(); ++r) {
        for (unsigned int c = 0; c < U.columns(); ++c) {
            if (r == c)
                ASSERT(U.at(r, c) == T{1});
            else
                ASSERT(U.at(r, c) == T{0});
        }
    }
    for (unsigned int r = 0; r < Vt.rows(); ++r) {
        for (unsigned int c = 0; c < Vt.columns(); ++c) {
            if (r == c)
                ASSERT(Vt.at(r, c) == T{1});
            else
                ASSERT(Vt.at(r, c) == T{0});
        }
    }
#endif
    // TODO for m >>= n, use a QR first to speed up this the bidiagonalization
    ::bidiagonalize(A, U, S, Vt);  // A = QSP^t
    ::diagonalize(U, S, Vt, std::forward<F>(condition));
#ifdef DEBUG
    TempMatrix<T, RowMajorIndexing> m{A.rows(), A.columns(), TEMPMEM};
    TempMatrix<T, RowMajorIndexing> m2{A.rows(), A.columns(), TEMPMEM};
    ::mul(U.cref(), S.cref(), m.ref());
    ::mul(m.cref(), Vt.cref(), m2.ref());
    ::subs(m2.ref(), A);
    auto s = ::matrixSquareSum(m2.cref());
    checkZero(s);
#endif
    REGISTER_ERROR(error(U.cref(), S.cref(), Vt.cref()));
}

template <typename T, typename I>
T error(const ConstMatrixRef<T, I>& m) {
    T ret{0};
    for (unsigned int r = 0; r < m.rows(); ++r)
        for (unsigned int c = 0; c < m.columns(); ++c)
            ret = std::max(ret, error(m.at(r, c)));
    return ret;
}
