#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

#include "algorithms/findroot.hpp"
#include "matrix.hpp"

template <typename T, typename V = T>
class Polynomial
{
 public:
    Polynomial() : data{static_cast<T>(0)} {}
    explicit Polynomial(const std::vector<T>& v) : data(v) {}
    explicit Polynomial(std::vector<T>&& v) : data(std::move(v)) {}

    T operator()(const V& x) const {
        if (data.size() == 0)
            return T{0};
        T ret{0};
        for (unsigned int i = static_cast<unsigned int>(data.size()); i > 0; --i) {
            ret = ret * x;
            ret += data[i - 1];
        }
        return ret;
    }

    static Polynomial roots(const std::vector<T>& z) {
        std::vector<T> values(z.size() + 1, T{0});
        fillRoots(z, values);
        return Polynomial(std::move(values));
    }

    void findRoots(std::vector<T>& x) const {
        if (data.size() < 2)
            return;
        if (data.size() == 2) {
            x.resize(1);
            ::find_roots(data[1], data[0], x[0]);
        }
        else if (data.size() == 3) {
            x.resize(2);
            ::find_roots(data[2], data[1], data[0], x[1], x[0]);
        }
        else if (data.size() == 4) {
            x.resize(3);
            ::find_roots(data[3], data[2], data[1], data[0], x[2], x[1], x[0]);
        }
        else
            ASSERT(false);
    }

    Polynomial derivative(unsigned int d = 1) const {
        if (data.size() <= 1)
            return Polynomial();
        std::vector<T> values(data.size() - d);
        for (unsigned int i = 0; i < values.size(); ++i)
            values[i] = data[i + d] * static_cast<V>(derivativeProd(i + d, d));
        return Polynomial(std::move(values));
    }

    struct Sample
    {
        V place;
        T value;
        unsigned int derivative;
    };
    static Polynomial samples(const std::vector<Sample>& samples, const unsigned int degree) {
        const unsigned int numSamples = static_cast<unsigned int>(samples.size());
        ASSERT(samples.size() > 0);
        std::vector<V> vA(numSamples * (degree + 1));
        std::vector<T> vb(numSamples);
        std::vector<T> vx(degree + 1);
        MatrixRef<V, RowMajorIndexing> A{vA.data(), numSamples, degree + 1};
        MatrixRef<T, ColumnMajorIndexing> b{vb.data(), numSamples, 1};
        MatrixRef<T, ColumnMajorIndexing> x{vx.data(), degree + 1, 1};
        for (unsigned int r = 0; r < numSamples; ++r) {
            b.at(r, 0u) = samples[r].value;
            for (unsigned int c = 0; c < samples[r].derivative; ++c)
                A.at(r, c) = V{0};
            for (unsigned int c = samples[r].derivative; c < A.columns(); ++c)
                A.at(r, c) = ::pow(samples[r].place, c - samples[r].derivative)
                             * static_cast<V>(derivativeProd(c, samples[r].derivative));
        }
        if (vx.size() == vb.size())
            ::solve(A.cref(), x, b.cref());
        else
            ::approximate(A.cref(), x, b.cref());
        return Polynomial(std::move(vx));
    }

    friend std::ostream& operator<<(std::ostream& out, const Polynomial& p) {
        for (unsigned int i = p.data.size(); i > 0; --i) {
            out << 'x';
            if (i > 2)
                out << '^' << i;
            out << '*' << p.data[i];
            if (i > 1)
                out << ' ';
        }
        return out;
    }

 private:
    std::vector<T> data;

    static void fillRoots(const std::vector<T>& z, std::vector<T>& values, const T& prod = T{1},
                          unsigned int num = 0, unsigned int ind = 0) {
        if (ind == z.size()) {
            values[num] += prod;
        }
        else {
            fillRoots(z, values, -prod * z[ind], num, ind + 1);  // pick z_ind
            fillRoots(z, values, prod, num + 1, ind + 1);        // pick x
        }
    }

    static V derivativeProd(unsigned int n, unsigned int d) {
        // TODO this product could be improved
        V prod{1};
        for (unsigned int k = n - d + 1; k <= n; ++k)
            prod *= static_cast<V>(k);
        return prod;
    }
};
