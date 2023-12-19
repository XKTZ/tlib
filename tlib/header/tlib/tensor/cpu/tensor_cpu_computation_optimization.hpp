//
// Created by xktz on 12/11/23.
//

#ifndef __TLIB_TENSOR_CPU_COMPUTATION_OPTIMIZATION_HPP__
#define __TLIB_TENSOR_CPU_COMPUTATION_OPTIMIZATION_HPP__

#include "tlib/tlib_config.hpp"

#if _TLIB_OPTIMIZE_CPU_BLAS

#include "tlib/compute/tlib_engine.hpp"
#include "tlib/tensor/cpu/tensor_cpu.hpp"
#include "tlib/tensor/cpu/tensor_cpu_computation.hpp"
#include "tlib/tensor/cpu/tensor_cpu_computation_base.hpp"
#include "tlib/tensor/tensor_computational.hpp"
#include <cassert>
#include <utility>

namespace tensorlib {
namespace tensor {

namespace computational {

/**
 * This is a optimized float class for Computational<T,CPU> using OpenBLAS
 * please refer to general Computational's document
 */
template <>
struct Computational<float, device::CPU> {
private:
    using T = float;
    using Engine = compute::Engine<T>;

public:
    using Base = cpu::TensorBase<T>;

private:
    using Shape = tensorlib::base::Shape;
    using Index = tensorlib::base::Index;
    using ComputeBase = computational_cpu::ComputeBase<T, Base>;

    template <typename... Args>
        requires((computational_cpu::IsT<T, Args> || computational_cpu::IsBase<T, Args>) && ...)
    static bool isContiguousable(Args &&...args) {
        if (([](auto &&x) {
                if constexpr (computational_cpu::IsBase<T, decltype(x)>) {
                    return (x.isContiguous() || x.getTotalSize() >= BLAS_CONTIGUOUS);
                } else {
                    return true;
                }
            }(std::forward<Args>(args)) &&
             ...)) {
            (
                [](auto &&x) {
                    if constexpr (computational_cpu::IsBase<T, decltype(x)>) {
                        x.contiguous();
                    }
                }(args),
                ...);
            return true;
        }
        return false;
    }

    template <bool Commutative = false, typename A, typename B>
    static void requireDimensionBinaryOperable(A &&a, B &&b) {
        if constexpr (computational_cpu::IsPureBinary<T, A, B> && SHAPE_CHECK) {
            if constexpr (Commutative) {
                if (a.dimensionSize() > b.dimensionSize()) {
                    ComputeBase ::requireDimensionBinaryOperable(a.size(), b.size());
                } else {
                    ComputeBase ::requireDimensionBinaryOperable(b.size(), a.size());
                }
            } else {
                ComputeBase ::requireDimensionBinaryOperable(a.size(), b.size());
            }
        }
    }

public:
    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base negate(A &&a) {
        if (isContiguousable(a)) {
            return ComputeBase ::PointerLevel::unaryOperation(a, [](SizeType N, T *to, T *x) {
                Engine::axpy({N, to, 1}, -ComputationConstant<T>::One, {N, x, 1});
            });
        }
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return -x; });
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base mulInv(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(
            a, [](const auto &x) { return ComputationConstant<T>::One / x; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base add(A &&a, B &&b) {
        if (isContiguousable(a, b)) {
            requireDimensionBinaryOperable<true>(a, b);
            return ComputeBase::PointerLevel::binaryEvaluateOperation(
                a, b,
                [](SizeType N, T *to, T *x, T *y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, x, 1});
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, y, 1});
                },
                [](SizeType N, T *to, T *x, T y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, x, 1});
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &y, 0});
                },
                [](SizeType N, T *to, T x, T *y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &x, 0});
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, y, 1});
                });
        }
        return ComputeBase::template binaryEvaluateDistributor<A, B, true>(
            std::forward<A>(a), std::forward<B>(b), [](const auto &x, const auto &y) { return x + y; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void addEqual(A &&a, B &&b) {
        if (isContiguousable(a, b)) {
            requireDimensionBinaryOperable(a, b);
            ComputeBase::PointerLevel::binaryAssignmentOperation(
                a, b,
                [](SizeType N, T *to, T *x) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, x, 1});
                },
                [](SizeType N, T *to, T x) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &x, 0});
                });
        } else {
            ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                              [](auto &x, const auto &y) { x += y; });
        }
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base minus(A &&a, B &&b) {
        if (isContiguousable(a, b)) {
            requireDimensionBinaryOperable(a, b);
            return ComputeBase::PointerLevel::binaryEvaluateOperation(
                a, b,
                [](SizeType N, T *to, T *x, T *y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, x, 1});
                    Engine::axpy({N, to, 1}, -ComputationConstant<T>::One, {N, y, 1});
                },
                [](SizeType N, T *to, T *x, T y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, x, 1});
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &y, 0});
                },
                [](SizeType N, T *to, T x, T *y) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &x, 0});
                    Engine::axpy({N, to, 1}, -ComputationConstant<T>::One, {N, y, 1});
                });
        }
        return ComputeBase::template binaryEvaluateDistributor(std::forward<A>(a), std::forward<B>(b),
                                                               [](const auto &x, const auto &y) { return x - y; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void minusEqual(A &&a, B &&b) {
        if (isContiguousable(a, b)) {
            if constexpr (computational_cpu::IsPureBinary<T, A, B> && SHAPE_CHECK) {
                ComputeBase ::requireDimensionBinaryOperable(a.size(), b.size());
            }
            ComputeBase::PointerLevel::binaryAssignmentOperation(
                a, b,
                [](SizeType N, T *to, T *x) {
                    Engine::axpy({N, to, 1}, -ComputationConstant<T>::One, {N, x, 1});
                },
                [](SizeType N, T *to, T x) {
                    Engine::axpy({N, to, 1}, ComputationConstant<T>::One, {N, &x, 0});
                });
        } else {
            ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                              [](auto &x, const auto &y) { x -= y; });
        }
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base mul(A &&a, B &&b) {
        if constexpr (computational_cpu::IsT<T, B>) {
            if (isContiguousable(a)) {
                return ComputeBase::PointerLevel::semiBinaryEvaluateOperation(a, b, [](SizeType N, T *to, T *a, T x) {
                    Engine::axpy({N, to, 1}, x, {N, a, 1});
                });
            }
        } else if constexpr (computational_cpu::IsT<T, A>) {
            if (isContiguousable(b)) {
                return ComputeBase::PointerLevel::semiBinaryEvaluateOperation(a, b, [](SizeType N, T *to, T x, T *a) {
                    Engine::axpy({N, to, 1}, x, {N, a, 1});
                });
            }
        }
        return ComputeBase::template binaryEvaluateDistributor<A, B, true>(
            std::forward<A>(a), std::forward<B>(b), [](const auto &x, const auto &y) { return x * y; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void mulEqual(A &&a, B &&b) {
        if constexpr (computational_cpu::IsT<T, B>) {
            if (isContiguousable(a)) {
                ComputeBase::PointerLevel::semiBinaryFunctionAssignOperation(a, b, [](SizeType N, T *to, T x) {
                    Engine::scal({N, to, 1}, x);
                });
                return;
            }
        }
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x *= y; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base div(A &&a, B &&b) {
        if constexpr (computational_cpu::IsT<T, B>) {
            if (isContiguousable(a)) {
                return ComputeBase ::PointerLevel::semiBinaryEvaluateOperation(
                    a, b, [d = ComputationConstant<T>::One / b](SizeType N, T *to, T *a, T _) {
                        Engine::axpy({N, to, 1}, d, {N, to, 1});
                    });
            }
        }
        return ComputeBase::template binaryEvaluateDistributor(std::forward<A>(a), std::forward<B>(b),
                                                               [](const auto &x, const auto &y) { return x / y; });
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void divEqual(A &&a, B &&b) {
        if constexpr (computational_cpu::IsT<T, B>) {
            if (isContiguousable(a)) {
                ComputeBase::PointerLevel::semiBinaryFunctionAssignOperation(
                    a, b, [d = ComputationConstant<T>::One / b](SizeType N, T *to, T _) {
                        Engine::scal({N, to, 1}, d);
                    });
                return;
            }
        } else {
            ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                              [](auto &x, const auto &y) { x /= y; });
        }
    }

    template <typename A, typename B>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void setEqual(A &&a, B &&b) {
        if (isContiguousable(a, b)) {
            ComputeBase::PointerLevel::binaryAssignmentOperation(
                a, b,
                [](SizeType N, T *to, T *x) {
                    Engine::copy({N, to, 1}, {N, x, 1});
                },
                [](SizeType N, T *to, T x) {
                    Engine::copy({N, to, 1}, {N, &x, 0});
                });
        }
        ComputeBase::template binaryAssignmentDistributor(std::forward<A>(a), std::forward<B>(b),
                                                          [](auto &x, const auto &y) { x = y; });
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base exp(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::exp(x); });
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base log(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::log(x); });
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base abs(A &&a) {
        return ComputeBase::template unaryEvaluateOperation(a, [](const auto &x) { return std::abs(x); });
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base sum(A &&a) {
        if (isContiguousable(a)) {
            Base result = Base::ofShape();
            T tmp = ComputationConstant<T>::One;
            (*result.raw()) = Engine::dot({a.getTotalSize(), a.raw(), 1}, {a.getTotalSize(), &tmp, 0});
            return result;
        }
        return ComputeBase::template dimensionalOperation(
            a, Index::generateIncreasingSequence(a.dimensionSize()), [](auto &x, const auto &y) { x = x + y; },
            ComputationConstant<T>::Zero);
    }

    template <typename A>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base sum(A &&a, const Index &dim) {
        if (([&dim, N = a.dimensionSize()]() {
                for (SizeType i = 0, n = dim.size(); i < n; i++) {
                    if (!(n + dim[i] == N + i)) { // dim[i] == N - n + i
                        return false;
                    }
                }
                return true;
            }()) &&
            isContiguousable(a)) {
            auto &aShape = a.size();
            Shape toShape(aShape.size() - dim.size());
            for (SizeType i = 0, n = toShape.size(); i < n; i++) {
                toShape[i] = aShape[i];
            }
            Base to = Base::ofShape(toShape);
            SizeType sz = 1;
            for (auto x : dim) {
                sz *= aShape[x];
            }
            T *tmpdot = new T[sz];
            T tmp = ComputationConstant<T>::One;
            Engine::copy({sz, tmpdot, 1}, {sz, &tmp, 0});

            Engine::gemv(compute::ComputeMatrix<T>::ROW, ComputationConstant<T>::Zero, {to.getTotalSize(), to.raw(), 1},
                         ComputationConstant<T>::One,
                         {compute::ComputeMatrix<T>::NO_TRANS, to.getTotalSize(), sz, a.raw(), sz}, {sz, tmpdot, 1});

            delete[] tmpdot;
            return to;
        }
        return ComputeBase::template dimensionalOperation(
            a, dim, [](auto &x, const auto &y) { x = x + y; }, ComputationConstant<T>::Zero);
    }

    template <typename A, typename B>
        requires computational_cpu::IsPureBinary<T, A, B>
    static Base matMul(A &&a, B &&b) {
        if constexpr (SHAPE_CHECK) {
            auto &aShape = a.size(), &bShape = b.size();
            if (!(ComputeBase::isMatrixMultiplicationOperable(aShape, bShape))) {
                throw std::out_of_range(tensorlib::base::shapeToString(aShape) + " x " +
                                        tensorlib::base::shapeToString(bShape) + " is not a valid matrix mul shape");
            }
        }

        if (isContiguousable(a, b)) {
            auto &aShape = a.size(), &bShape = b.size();

            Shape toShape(aShape.size());
            SizeType n = aShape[aShape.size() - 2], m = aShape.back(), k = bShape.back();
            SizeType nums = 1;
            auto &shpLarger = aShape.size() > bShape.size() ? aShape : bShape;
            for (SizeType i = 0, T = shpLarger.size(); i + 2 < T; i++)
                nums *= shpLarger[i], toShape[i] = shpLarger[i];
            toShape[toShape.size() - 2] = n;
            toShape.back() = k;

            Base to = Base::ofShape(toShape);

            if (bShape.size() == 2) {
                // (x1, x2, ..., n, m) * (m, k)
                auto pto = to.raw();
                auto pa = a.raw();
                auto pb = b.raw();

                //                cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE ::CblasNoTrans,
                //                CBLAS_TRANSPOSE::CblasNoTrans,
                //                            blasint(nums * n), blasint(k), blasint(m), ComputationConstant<T>::One,
                //                            pa, blasint(m), pb, blasint(k), ComputationConstant<T>::Zero, pto,
                //                            blasint(k));

                Engine::gemm(compute::ComputeMatrix<T>::ROW, ComputationConstant<T>::Zero,
                             {compute::ComputeMatrix<T>::NO_TRANS, nums * n, k, pto, k}, ComputationConstant<T>::One,
                             {compute::ComputeMatrix<T>::NO_TRANS, nums * n, m, pa, m},
                             {compute::ComputeMatrix<T>::NO_TRANS, m, k, pb, k});

            } else {
                ComputeBase ::PointerLevel::multinaryOperation(
                    nums * n * k, n * k,
                    [n, m, k](SizeType ignore, T *to, T *a, T *b) {
                        //                        cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans,
                        //                                    CBLAS_TRANSPOSE::CblasNoTrans, blasint(n), blasint(k),
                        //                                    blasint(m), ComputationConstant<T>::One, a, blasint(m), b,
                        //                                    blasint(k), ComputationConstant<T>::Zero, to, blasint(k));
                        Engine::gemm(compute::ComputeMatrix<T>::ROW, ComputationConstant<T>::Zero,
                                     {compute::ComputeMatrix<T>::NO_TRANS, n, k, to, k}, ComputationConstant<T>::One,
                                     {compute::ComputeMatrix<T>::NO_TRANS, n, m, a, m},
                                     {compute::ComputeMatrix<T>::NO_TRANS, m, k, b, k});
                    },
                    std::make_pair(to.raw(), std::make_tuple(n * k, SizeType(0), to.getTotalSize())),
                    std::make_pair(a.raw(), std::make_tuple(n * m, SizeType(0), a.getTotalSize())),
                    std::make_pair(b.raw(), std::make_tuple(m * k, SizeType(0), b.getTotalSize())));
            }

            return to;
        }

        return ComputeBase::template matrixMultiplyDistributor(std::forward<A>(a), std::forward<B>(b));
    }

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

        a.contiguous();
        b.contiguous();

        if (isContiguousable(a, b)) {
            auto &aShape = a.size(), &bShape = b.size();
            SizeType Batch = aShape[0];
            SizeType C = aShape[1], H = aShape[2], W = aShape[3];
            SizeType D = bShape[0], X = bShape[2], Y = bShape[3];
            SizeType oH = H - X + 1, oW = W - Y + 1;

            Base result = Base::ofShape({Batch, D, oH, oW});

            T *pa = a.raw(), *pb = b.raw(), *pres = result.raw();

            T *matConv = new T[(C * X * Y) * (oH * oW) * Batch];

            for (SizeType c = 0; c < C; c++) {
                for (SizeType x = 0; x < X; x++) {
                    for (SizeType y = 0; y < Y; y++) {
                        // ker: [oH * oW * B]
                        T *ker = matConv + (c * X * Y + x * Y + y) * Batch * oH * oW;
                        for (SizeType batch = 0; batch < Batch; batch++) {
                            // [H, W]
                            T *pac = pa + batch * C * H * W + c * H * W;

                            for (SizeType i = 0; i < oH; i++) {
                                // cblas_scopy(oW, pac + (i + x) * W + y, 1, ker + Batch * (i * oW), Batch);
                                Engine::copy({oW, pac + (i + x) * W + y, 1}, {oW, ker + Batch * (i + oW), Batch});

                                /*for (SizeType j = 0; j < oW; j++) {
                                    ker[Batch * (i * oW + j)] = pac[(i + x) * W + (j + y)];
                                }*/
                            }

                            ker++;
                        }
                    }
                }
            }

            // cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE ::CblasNoTrans, CBLAS_TRANSPOSE ::CblasNoTrans,
            // D,
            //            oH * oW * Batch, C * X * Y, ComputationConstant<T>::One, pb, C * X * Y, matConv,
            //            oH * oW * Batch, ComputationConstant<T>::Zero, pres, oH * oW * Batch);

            Engine::gemm(compute::ComputeMatrix<T>::ROW, ComputationConstant<T>::Zero,
                         {compute::ComputeMatrix<T>::NO_TRANS, D, oH * oW * Batch, pres, oH * oW * Batch},
                         ComputationConstant<T>::One,
                         {compute::ComputeMatrix<T>::NO_TRANS, D, C * X * Y, pb, C * X * Y},
                         {compute::ComputeMatrix<T>::NO_TRANS, C * X * Y, oH * oW * Batch, matConv, oH * oW * Batch});

            // now, pres is (D * oH * oW) * Batch
            // transpose

            // cblas_simatcopy(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans, D * oH * oW, Batch,
            //                ComputationConstant<T>::One, pres, Batch, D * oH * oW);
            Engine::imatcopy(compute::ComputeMatrix<T>::ROW,
                             {compute::ComputeMatrix<T>::TRANS, D * oH * oW, Batch, pres, Batch},
                             ComputationConstant<T>::One, D * oH * oW);

            delete[] matConv;

            return result;
        }

        return ComputeBase::template conv2d(std::forward<A>(a), std::forward<B>(b));
    }

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

    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base transpose(A &&a, SizeType d1, SizeType d2) {
        if constexpr (SHAPE_CHECK) {
            if (a.dimensionSize() < 2 || d1 >= a.dimensionSize() || d2 >= a.dimensionSize()) {
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

    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base apply(A &&a, auto &&f) {
        return ComputeBase::template unaryEvaluateOperation(a, f);
    }

    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static void forEach(A &&a, auto &&f) {
        ComputeBase::template unaryFunctionAssignOperation(a, f);
    }

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

    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base reshape(A &&a, const Shape &shp) {
        return a.reshape(shp);
    }

    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base permute(A &&a, const Index &idx) {
        return a.permute(idx);
    }

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
            Shape x = b[0]->size();
            for (SizeType i = 1; i < b.size(); i++) {
                if (x != b[i]->size()) {
                    throw std::out_of_range("Not concattable");
                }
            }
        }
        auto &origShp = b[0]->size();
        Shape toShape = Shape(origShp.size() + 1);
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

#endif

#endif
