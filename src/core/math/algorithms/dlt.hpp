#pragma once

#include <vector>

#include "matrixref.hpp"
#include "projective2d.hpp"
#include "projective3d.hpp"
#include "types.hpp"

// this is the Direct Linear Transformations algorithm

template <typename T, bool Temp = true>
class EquationBuilder
{
 public:
    using ValueType = T;
    struct Config
    {};
    using ConfigType = Config;

    bool enough() const { return A.rows() >= requiredEqs; }

 protected:
    explicit EquationBuilder(const ConfigType&, unsigned int maxEquations, unsigned int numVars,
                             unsigned int numRequiredEquations = 0)
      : A(maxEquations, numVars),
        b(maxEquations, 1u),
        requiredEqs(numRequiredEquations == 0 ? numVars : numRequiredEquations) {}

    template <typename I>
    bool solveSimple(const MatrixRef<T, I>& res) const& {
        ASSERT(A.rows() >= A.columns());
        ASSERT(A.rows() == b.rows());
        if (A.rows() == A.columns())
            ::solve(A.cref(), res, b.cref());
        else
            ::approximate(A.cref(), res, b.cref());
        return true;  // TODO 0-division maybe?
    }

    template <typename I>
    bool solveSimple(const MatrixRef<T, I>& res) && {
        ASSERT(A.rows() >= A.columns());
        ASSERT(A.rows() == b.rows());
        if (A.rows() == A.columns())
            ::solve(std::move(A.ref()), res, b.cref());
        else
            ::approximate(std::move(A.ref()), res, b.cref());
        return true;  // TODO 0-division maybe?
    }

    template <bool Temp2>
    void getNullSpace(DynMatrix<T, Temp2>& res, unsigned int forceCount = 0u) const& {
        ::kernel_right(A.cref(), res, forceCount);
    }

    template <bool Temp2>
    void getNullSpace(DynMatrix<T, Temp2>& res, unsigned int forceCount = 0u) && {
        ::kernel_right(A, res, forceCount);
    }

    T* addEquation(const T& value) {
        *b.addRow() = value;
        return A.addRow();
    }

    T* addEquation(T&& value) {
        *b.addRow() = std::move(value);
        return A.addRow();
    }

    T* addEquation() {
        // The solution will only be possible with null space
        return A.addRow();
    }

    void reset() {
        A.reset();
        b.reset();
    }

 private:
    DynMatrix<T, Temp> A;
    DynMatrix<T, Temp> b;
    unsigned int requiredEqs;
};

template <typename T, unsigned int R, unsigned int C, bool Temp = true,
          typename I = ColumnMajorIndexing>
class MatrixBuilder : public EquationBuilder<T, Temp>
{
 private:
    static const unsigned int maxEquationsPerData = R;
    static const unsigned int maxEquationsForHomogene = 1;

 public:
    struct Config
    {};
    using ConfigType = Config;
    explicit MatrixBuilder(const ConfigType&, unsigned int maxNumPoints, bool homogeneous,
                           const I& _indexing = {})
      : EquationBuilder<T>(
        typename EquationBuilder<T>::ConfigType{},
        maxEquationsPerData * maxNumPoints + maxEquationsForHomogene * homogeneous, R * C),
        indexing(_indexing) {}

    explicit MatrixBuilder(const ConfigType&, unsigned int maxNumPoints, bool homogeneous,
                           unsigned int numRequiredEquations, const I& _indexing = {})
      : EquationBuilder<T>(
        typename EquationBuilder<T>::ConfigType{},
        maxEquationsPerData * maxNumPoints + maxEquationsForHomogene * homogeneous, R * C,
        numRequiredEquations),
        indexing(_indexing) {}

    bool solveSimple(const MatrixRef<T, I>& ref) const& {
        ASSERT(ref.rows() == R && ref.columns() == C);
        MatrixRef<T, ColumnMajorIndexing> m(ref.getData(), R * C, 1u);
        return EquationBuilder<T>::solveSimple(m);
    }

    bool solveSimple(const MatrixRef<T, I>& ref) && {
        ASSERT(ref.rows() == R && ref.columns() == C);
        MatrixRef<T, ColumnMajorIndexing> m(ref.getData(), R * C, 1u);
        return std::move(*this).EquationBuilder<T>::solveSimple(m);
    }

    template <typename... I2>
    bool getNullSpace(unsigned int forceCount, const MatrixRef<T, I2>&... ref) const& {
        DynMatrix<T, true> res{R * C, R * C};
        EquationBuilder<T>::getNullSpace(res, forceCount);
        return extractRes(res, ref...);
    }

    template <typename... I2>
    bool getNullSpace(unsigned int forceCount, const MatrixRef<T, I2>&... ref) && {
        DynMatrix<T, true> res{R * C, R * C};
        std::move(*this).EquationBuilder<T>::getNullSpace(res, forceCount);
        return extractRes(res, ref...);
    }

    template <typename V, typename V2>
    void matchInhomogeneous(const V& original, const V2& transformed, const T& weight = T{1}) {
        for (unsigned int r = 0; r < R; ++r) {
            T* row = EquationBuilder<T>::addEquation(transformed.at(r, 0u) * weight);
            for (unsigned int i = 0; i < R * C; ++i)
                row[i] = T{0};
            for (unsigned int c = 0; c < C; ++c)
                row[indexing(r, c, R, C)] = original.at(c, 0u) * weight;
        }
    }

    template <typename V, typename V2>
    void matchHomogeneous(const V& original, const V2& transformed, const T& weight = T{1}) {
        // Ha * a'.h = a' * (Ha).h

        // (Ha)i * a'.h = a'i * (Ha).h   (i : [1..R-1])
        // (Hi1*a1 + ... + HiC*aC) * a'.h = a'i * (Ha).h   (i : [1..R-1])
        // (Hi1*a1 + ... + HiC*aC) * a'R = a'i * (Ha).h   (i : [1..R-1])
        // (Hi1*a1 + ... + HiC*aC) * a'R = a'i * (HR1*a1 + ... + HRC*aC)   (i : [1..R-1])
        // Hi1*a1*a'R + ... + HiC*aC*a'R = a'i*HR1*a1 + ... + a'i*HRC*aC   (i : [1..R-1])
        // Hi1*a1*a'R + ... + HiC*aC*a'R - a'i*HR1*a1 - ... - a'i*HRC*aC = 0   (i : [1..R-1])

        for (unsigned int r = 0; r < R - 1; ++r) {
            T* row = EquationBuilder<T>::addEquation(T{0});
            for (unsigned int i = 0; i < R * C; ++i)
                row[i] = T{0};
            for (unsigned int c = 0; c < C; ++c) {
                row[indexing(r, c, R, C)] = original.at(c, 0u) * transformed.at(R - 1, 0u) * weight;
                row[indexing(R - 1, c, R, C)] =
                  -original.at(c, 0u) * transformed.at(r, 0u) * weight;
            }
        }
    }

    void set(const T& value, unsigned int r, unsigned int c, const T& weight = T{1}) {
        T* row = EquationBuilder<T>::addEquation(value * weight);
        for (unsigned int i = 0; i < R * C; ++i)
            row[i] = T{0};
        row[indexing(r, c, R, C)] = weight;
    }

 protected:
    const I& getIndexing() const { return indexing; }

 private:
    I indexing;

    template <bool Temp2, typename I2>
    static void extractHelper(const DynMatrix<T, Temp2>& res, unsigned int ind,
                              const MatrixRef<T, I2>& ref) {
        ASSERT(ref.rows() == R && ref.columns() == C);
        MatrixRef<T, ColumnMajorIndexing> m(ref.getData(), R * C, 1u);
        auto resRef = res.cref();
        for (unsigned int r = 0; r < m.rows(); ++r)
            m.at(r, 0u) = resRef.at(ind, r);
    }

    template <bool Temp2, typename... I2>
    static bool extractRes(const DynMatrix<T, Temp2>& res, const MatrixRef<T, I2>&... ref) {
        if (sizeof...(ref) != res.rows())
            return false;
        unsigned int ind = 0;
        int unused[]{(extractHelper(res, ind++, ref), 1)...};
        static_cast<void>(unused);
        return true;
    }
};

template <typename T, unsigned int R, unsigned int C, bool Temp = true,
          typename I = ColumnMajorIndexing>
class ProjectionBuilder : public MatrixBuilder<T, R, C, Temp, I>
{
 public:
    using SolutionType = MatrixT<T, R, C>;
    struct Config
    {
        unsigned int r;
        unsigned int c;
        T value;
        T weight = T{1};
    };
    using ConfigType = Config;
    explicit ProjectionBuilder(const ConfigType& _config,
                               unsigned int maxNumPoints = (R * C - 1u) / (R - 1u),
                               const I& indexing = {})
      : MatrixBuilder<T, R, C, Temp, I>(typename MatrixBuilder<T, R, C, Temp, I>::ConfigType{},
                                        maxNumPoints, true, indexing),
        config(_config) {
        MatrixBuilder<T, R, C, Temp, I>::set(T{1}, R - 1u, C - 1u);
    }

    template <typename V, typename V2>
    void match(const V& original, const V2& transformed, const T& weight = T{1}) {
        MatrixBuilder<T, R, C, Temp, I>::matchHomogeneous(original, transformed, weight);
    }

    bool solveSimple(SolutionType& solution) const& {
        return MatrixBuilder<T, R, C, Temp, I>::solveSimple(solution.ref());
    }

    bool solveSimple(SolutionType& solution) && {
        return std::move(*this).MatrixBuilder<T, R, C, Temp, I>::solveSimple(solution.ref());
    }

    void reset() {
        EquationBuilder<T, Temp>::reset();
        MatrixBuilder<T, R, C, Temp, I>::set(config.value, config.r, config.c, config.weight);
    }

 private:
    ConfigType config;
};

// 8-point algorithm
template <typename T, bool Temp = true, typename IterCond = IterationCondition<T>,
          typename I = ColumnMajorIndexing>
class FundamentalMatrixBuilder : public MatrixBuilder<T, 3u, 3u, Temp, I>
{
 public:
    using SolutionType = MatrixT<T, 3u, 3u>;
    using ConfigType = IterCond;
    explicit FundamentalMatrixBuilder(const ConfigType& _config, unsigned int maxNumPoints = 8u,
                                      const I& indexing = {})
      : MatrixBuilder<T, 3u, 3u, Temp, I>(typename MatrixBuilder<T, 3u, 3u, Temp, I>::ConfigType{},
                                          maxNumPoints, true, 8u, indexing),
        config(_config) {}

    template <typename V, typename V2>
    void match(const V& a, const V2& b, const T& weight = T{1}) {
        // dot(Ha, b) = 0

        T* row = EquationBuilder<T>::addEquation();
        for (unsigned int c = 0; c < 3u; ++c) {
            for (unsigned int r = 0; r < 3u; ++r) {
                const unsigned int ind =
                  MatrixBuilder<T, 3u, 3u, Temp, I>::getIndexing()(r, c, 3u, 3u);
                row[ind] = a.at(c, 0u) * b.at(r, 0u) * weight;
            }
        }
    }

    bool solveSimple(SolutionType& solution) const& {
        bool ret = MatrixBuilder<T, 3u, 3u, Temp, I>::getNullSpace(1u, solution.ref());
        if (ret)
            finalizeSimple(solution);
        return ret;
    }

    bool solveSimple(SolutionType& solution) && {
        bool ret =
          std::move(*this).MatrixBuilder<T, 3u, 3u, Temp, I>::getNullSpace(1u, solution.ref());
        if (ret)
            finalizeSimple(solution);
        return ret;
    }

    void reset() { EquationBuilder<T, Temp>::reset(); }

 private:
    ConfigType config;

    void finalizeSimple(SolutionType& solution) const {
        using std::abs;
        SolutionType U, D, V;
        ::SVD(solution.cref(), U.ref(), D.ref(), V.ref(), config);
        unsigned int min = 0;
        for (unsigned int i = 1; i < D.rows(); ++i)
            if (abs(D.at(i, i)) < abs(D.at(min, min)))
                min = i;
        D.at(min, min) = 0;
        const unsigned int a1 = (min + 1) % solution.rows();
        const unsigned int a2 = (min + 2) % solution.rows();
        // normalization
        using math::hypot;
        const T inv = T{1} / hypot(D.at(a1, a1), D.at(a2, a2));
        D.at(a1, a1) *= inv;
        D.at(a2, a2) *= inv;
        solution = U * D * V;
    }
};

// template <typename T>
// class Line2DBuilder final : public EquationBuilder<T>
// {
//  private:
//     static const unsigned int maxEquationsPerData = 1;

//  public:
//     using ValueType = Line2D<MatrixT<T, 3u, 1u>>;

//     explicit Line2DBuilder(unsigned int maxNumPoints)
//       : EquationBuilder<T>(maxEquationsPerData * maxNumPoints, 3) {}

//     ValueType solve() const& {
//         ValueType ret;
//         EquationBuilder<T>::solve(ret.matrix().ref());
//         return ret;
//     }

//     ValueType solve() && {
//         ValueType ret;
//         EquationBuilder<T>::solve(ret.matrix().ref());
//         return ret;
//     }

//     template <typename H, typename IH>
//     void match(const Point2D<H, IH>& point, const T& weight = T{1}) {
//         T* row = EquationBuilder<T>::addEquation(T{0});
//         row[0] = point.matrix().at(0u, 0u) * weight;
//         row[1] = point.matrix().at(1u, 0u) * weight;
//         row[2] = point.matrix().at(2u, 0u) * weight;
//     }

//     // TODO implement matching to a pointset

//  private:
// };
