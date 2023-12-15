#ifndef __TLIB_TENSOR_CPU_COMPUTATION_HPP__
#define __TLIB_TENSOR_CPU_COMPUTATION_HPP__

#include "tlib/tensor/cpu/tensor_cpu.hpp"
#include "tlib/tensor/cpu/tensor_cpu_computation_base.hpp"
#include "tlib/tensor/tensor_computational.hpp"

namespace tensorlib {
namespace tensor {

namespace computational {

/**
 * Computational for CPU
 * Notice many tparams are defined A & B under the class. Those are used for supporting both lr values of base
 * So it isn't documented in this file
 * @tparam T Type of variable
 */
template <typename T>
struct Computational<T, device::CPU> {
public:
    using Base = cpu::TensorBase<T>;

private:
    using Shape = tensorlib::base::Shape;
    using Index = tensorlib::base::Index;
    using ComputeBase = computational_cpu::ComputeBase<T, Base>;

public:

    /**
     * Negate a
     * @param a a
     * @return -a
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base negate(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return -x; });
    }

    /**
     * Calculate multiplicative inverse of a
     * @param a a
     * @return 1/a
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base mulInv(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(
            a, [](const auto &x) { return ComputationConstant<T>::One / x; });
    }

    /**
     * Add a and b
     * @param a a
     * @param b b
     * @return a + b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base add(A &&a, B &&b) {
        return ComputeBase::template binaryEvaluateDistributor<A, B, true>(
            std::forward<A>(a), std::forward<B>(b), [](const auto &x, const auto &y) { return x + y; });
    }

    /**
     * a += b
     * @param a
     * @param b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void addEqual(A &&a, B &&b) {
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x += y; });
    }

    /**
     * minus a and b
     * @param a a
     * @param b b
     * @return a - b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base minus(A &&a, B &&b) {
        return ComputeBase::template binaryEvaluateDistributor(std::forward<A>(a), std::forward<B>(b),
                                                               [](const auto &x, const auto &y) { return x - y; });
    }

    /**
     * a -= b
     * @param a a
     * @param b b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void minusEqual(A &&a, B &&b) {
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x -= y; });
    }

    /**
     * times a and b
     * @param a a
     * @param b b
     * @return a * b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base mul(A &&a, B &&b) {
        return ComputeBase::template binaryEvaluateDistributor<A, B, true>(
            std::forward<A>(a), std::forward<B>(b), [](const auto &x, const auto &y) { return x * y; });
    }

    /**
     * a *= b
     * @param a a
     * @param b b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void mulEqual(A &&a, B &&b) {
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x *= y; });
    }

    /**
     * divide a and b
     * @param a a
     * @param b b
     * @return a / b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base div(A &&a, B &&b) {
        return ComputeBase::template binaryEvaluateDistributor(std::forward<A>(a), std::forward<B>(b),
                                                               [](const auto &x, const auto &y) { return x / y; });
    }

    /**
     * a /= b
     * @tparam A
     * @tparam B
     * @param a
     * @param b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void divEqual(A &&a, B &&b) {
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x /= y; });
    }

    /**
     * a := b
     * @param a a
     * @param b b
     */
    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void setEqual(A &&a, B &&b) {
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x = y; });
    }

    /**
     * @param a
     * @return exp(a)
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base exp(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::exp(x); });
    }

    /**
     * @param a
     * @return log(a)
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base log(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::log(x); });
    }

    /**
     * @param a
     * @return |a|
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base abs(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::abs(x); });
    }

    /**
     * @param a a
     * @return sum(a)
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base sum(A &&a) {
        return ComputeBase::template dimensionalOperation(
            a, Index::generateIncreasingSequence(a.dimensionSize()), [](auto &x, const auto &y) { x = x + y; },
            ComputationConstant<T>::Zero);
    }

    /**
     * @param a a
     * @param dim dim
     * @return sum(a) on dim
     */
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base sum(A &&a, const Index &dim) {
        return ComputeBase::template dimensionalOperation(
            a, dim, [](auto &x, const auto &y) { x = x + y; }, ComputationConstant<T>::Zero);
    }

    /**
     * @param a A
     * @param b B
     * @return AB
     */
    template <typename A, typename B>
        requires computational_cpu::IsPureBinary<T, A, B>
    static Base matMul(A &&a, B &&b) {
        auto aShape = a.size(), bShape = b.size();

        if constexpr (SHAPE_CHECK) {
            if (!(ComputeBase::isMatrixMultiplicationOperable(aShape, bShape))) {
                throw std::out_of_range(tensorlib::base::shapeToString(aShape) + " x " +
                                        tensorlib::base::shapeToString(bShape) + " is not a valid matrix mul shape");
            }
        }

        return ComputeBase::template matrixMultiplyDistributor(std::forward<A>(a), std::forward<B>(b));
    }

    /**
     * @param a a
     * @param b b
     * @return a conv2d b
     */
    template <typename A, typename B>
        requires computational_cpu::IsPureBinary<T, A, B>
    static Base conv2d(A &&a, B &&b) {
        if constexpr (SHAPE_CHECK) {
            auto &aShape = a.size(), &bShape = b.size();
            if (a.dimensionSize() != 4 || b.dimensionSize() != 4) {
                throw std::out_of_range(util::functional::messageOf("Shape ", base::shapeToString(aShape), " & ",
                                                                    base::shapeToString(bShape), " cannot conv2d"));
            }

            auto loc = a.dimensionSize() - 3;
            if (aShape[loc] != bShape[1] || aShape[loc + 1] < bShape[2] || aShape[loc + 2] < bShape[3]) {
                throw std::out_of_range(util::functional::messageOf("Shape ", base::shapeToString(aShape), " & ",
                                                                    base::shapeToString(bShape), " cannot conv2d"));
            }
        }
        return ComputeBase::template conv2d(std::forward<A>(a), std::forward<B>(b));
    }

    /**
     * @param a a
     * @return a.t
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base transpose(A &&a) {
        if constexpr (SHAPE_CHECK) {
            if (a.dimensionSize() < 2) {
                throw std::out_of_range(
                    (std::stringstream() << tensorlib::base::shapeToString(a.size()) << " not good for transpose")
                        .str());
            }
        }

        return transpose(a, a.dimensionSize() - 2, a.dimensionSize() - 1);
    }

    /**
     * tranpose two dimensions of a
     * @param a a
     * @param d1 d1
     * @param d2 d2
     * @return a tranposed d1 d2
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base transpose(A &&a, SizeType d1, SizeType d2) {
        if constexpr (SHAPE_CHECK) {
            if (d1 >= a.dimensionSize() || d2 >= a.dimensionSize()) {
                throw std::out_of_range((std::stringstream() << d1 << " & " << d2 << " is not a valid tranpose for "
                                                             << tensorlib::base::shapeToString(a.size()))
                                            .str());
            }
        }
        if (Base::isIndexered(a)) {
            auto shp = a.size();
            std::swap(shp[d1], shp[d2]);
            return Base(a, std::move(shp), [d1, d2](auto &&idx) {
                auto nidx = idx;
                std::swap(nidx[d1], nidx[d2]);
                return nidx;
            });
        } else {
            Base b = a;
            b.swapDimension(d1, d2);
            return b;
        }
    }

    /**
     * expand a with dimension choice
     * @param a a
     * @param dimChoice dim choice
     * @return expand a with dim choice
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base expand(A &&a, auto &&dimChoice) {
        auto &aShape = a.size();
        Shape nShape(aShape.size() + dimChoice.size());
        SizeType n = nShape.size();
        SizeType aIdxOn = 0;
        SizeType dSize = dimChoice.size(), dimChoiceIdxOn = 0;

        std::shared_ptr<bool[]> isFolded = std::shared_ptr<bool[]>(new bool[n]);

        for (SizeType i = 0; i < n; i++) {
            if (dimChoiceIdxOn < dSize && dimChoice[dimChoiceIdxOn].first == i) {
                nShape[i] = dimChoice[dimChoiceIdxOn].second;
                dimChoiceIdxOn++;
                isFolded[i] = true;
            } else {
                nShape[i] = aShape[aIdxOn];
                aIdxOn++;
                isFolded[i] = false;
            }
        }

        return Base(a, std::move(nShape), [aSize = aShape.size(), n, isFolded = std::move(isFolded)](auto &&idx) {
            Index nidx = Index(aSize);
            SizeType j = 0;
            for (SizeType i = 0; i < n; i++) {
                if (!isFolded[i]) {
                    nidx[j++] = idx[i];
                }
            }
            return nidx;
        });
    }

    /**
     * apply function f on a
     * @param a a
     * @param f f
     * @return f(a)
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base apply(A &&a, auto &&f) {
        return ComputeBase::template unaryEvaluateOperation(a, f);
    }

    /**
     * f(a), where f may not be pure
     * @param a a
     * @param f f
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static void forEach(A &&a, auto &&f) {
        ComputeBase::template unaryFunctionAssignOperation(a, f);
    }

    /**
     * @param a a
     * @param b b
     * @return max(a, b)
     */
    template <typename A, typename B>
        requires computational_cpu::IsBase<T, A> && computational_cpu::IsBase<T, B>
    static Base max(A &&a, B &&b) {
        if constexpr (SHAPE_CHECK) {
            if (a.size() != b.size()) {
                throw std::out_of_range(util::functional::messageOf("Shape with ", base::shapeToString(a.size()),
                                                                    " and ", base::shapeToString(b.size()),
                                                                    " is not able to take max"));
            }
        }
        return ComputeBase::binaryEvaluateOperation(a, b, [](auto &&x, auto &&y) { return x > y ? x : y; });
    }

    /**
     * @param a a
     * @param b b
     * @return a > b ? 1 : 0
     */
    template <typename A, typename B>
        requires computational_cpu::IsBase<T, A> && computational_cpu::IsBase<T, B>
    static Base maxMask(A &&a, B &&b) {
        if constexpr (SHAPE_CHECK) {
            if (a.size() != b.size()) {
                throw std::out_of_range(util::functional::messageOf("Shape with ", base::shapeToString(a.size()),
                                                                    " and ", base::shapeToString(b.size()),
                                                                    " is not able to take max"));
            }
        }
        return ComputeBase::binaryEvaluateOperation(a, b, [](auto &&x, auto &&y) {
            return x > y ? ComputationalConstant<T>::One : ComputationalConstant<T>::Zero;
        });
    }

    /**
     * reshape a with shp
     * @param a a
     * @param shp shp
     * @return reshaped a
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base reshape(A &&a, const Shape &shp) {
        return a.reshape(shp);
    }

    /**
     * permute a
     * @param a a
     * @param idx idx
     * @return permute a
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base permute(A &&a, const Index &idx) {
        return a.permute(idx);
    }

    /**
     * power a
     * @param a a
     * @param s power
     * @return a^s
     */
    template <typename A, typename S>
        requires computational_cpu::IsBase<T, A>
    static Base pow(A &&a, const S &s) {
        if constexpr (std::is_integral_v<std::decay<S>>) {
            if (s >= ComputationConstant<std::decay<S>>::Zero) {
                return apply(a, [s](auto &&v) { return ComputeBase::fastPow(v, s); });
            }
        }
        return apply(a, [s](auto &&v) { return std::pow(v, s); });
    }

    /**
     * flip a's dimension
     * @param a a
     * @param d d
     * @return fliped
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base flip(A &&a, SizeType d) {
        if constexpr (SHAPE_CHECK) {
            if (d >= a.dimensionSize()) {
                throw std::out_of_range(
                    util::functional::messageOf("Not flippable ", base::shapeToString(a.size()), " on ", d));
            }
        }

        auto shp = a.size();
        SizeType n = shp[d];

        return Base(a, std::move(shp), [d, n](const auto &idx) {
            auto nidx = idx;
            nidx[d] = n - nidx[d] - 1;
            return nidx;
        });
    }

    /**
     * pad a
     * @param a a
     * @param d d
     * @return padded
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base pad(A &&a, auto &&d) {
        if constexpr (SHAPE_CHECK) {
            if (a.dimensionSize() != d.size()) {
                throw std::out_of_range(
                    util::functional::messageOf("Not paddable ", base::shapeToString(a.size()), [&d]() {
                        std::string s = "(";
                        for (auto [p, q] : d) {
                            s += util::functional::messageOf("(", p, ",", q, "),");
                        }
                        if (s.back() == ',') {
                            s.pop_back();
                        }
                        s.push_back(')');
                        return s;
                    }()));
            }
        }

        return ComputeBase::template pad(a, d);
    }

    /**
     * cut a along dimension
     * @param a a
     * @param d d
     * @return cutted
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base cut(A &&a, auto &&d) {
        if constexpr (SHAPE_CHECK) {
            if (a.dimensionSize() != d.size()) {
                throw std::out_of_range(
                    util::functional::messageOf("Not cuttable ", base::shapeToString(a.size()), [&d]() {
                        std::string s = "(";
                        for (auto [p, q] : d) {
                            s += util::functional::messageOf("(", p, ",", q, "),");
                        }
                        if (s.back() == ',') {
                            s.pop_back();
                        }
                        s.push_back(')');
                        return s;
                    }()));
            }
            auto &aShape = a.size();
            SizeType n = a.dimensionSize();
            for (SizeType i = 0; i < n; i++) {
                if (!(d[i].first <= d[i].second && d[i].second <= aShape[i])) {
                    throw std::out_of_range(
                        util::functional::messageOf("Not cuttable ", base::shapeToString(a.size()), [&d]() {
                            std::string s = "(";
                            for (auto [p, q] : d) {
                                s += util::functional::messageOf("(", p, ",", q, "),");
                            }
                            if (s.back() == ',') {
                                s.pop_back();
                            }
                            s.push_back(')');
                            return s;
                        }()));
                }
            }
        }
        return ComputeBase::template cut(a, d);
    }

    static Base concat(const std::vector<Base *> &b) {
        if constexpr (SHAPE_CHECK) {
            if (b.empty()) {
                throw std::out_of_range("Empty vector not concattable");
            }
            auto &x = b[0]->size();
            for (SizeType i = 1; i < b.size(); i++) {
                if (x != b[i]->size()) {
                    throw std::out_of_range("Not concattable");
                }
            }
        }
        auto &origShp = b[0]->size();
        auto toShape = Shape(origShp.size() + 1);
        toShape[0] = b.size();
        for (SizeType i = 0; i < origShp.size(); i++) {
            toShape[i + 1] = origShp[i];
        }

        Base result = Base::ofShape(toShape);

        for (SizeType i = 0; i < b.size(); i++) {
            setEqual(result[{i}], *(b[i]));
        }

        return result;
    }
};
}; // namespace computational
}; // namespace tensor
}; // namespace tensorlib

#include "tlib/tensor/cpu/tensor_cpu_computation_optimization.hpp"

#endif