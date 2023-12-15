#ifndef __TLIB_TENSOR_CPU_COMPUTATION_BASE_HPP__
#define __TLIB_TENSOR_CPU_COMPUTATION_BASE_HPP__

#include "tlib/tensor/tensor.hpp"
#include "tlib/tensor/tensor_computational.hpp"
#include <numeric>
#include <tuple>

namespace tensorlib {
namespace tensor {
namespace computational {

namespace computational_cpu {

/**
 * Is S a base for T
 * @tparam T T
 * @tparam S S
 */
template <typename T, typename S>
concept IsBase = std::is_same_v<std::decay_t<S>, cpu::TensorBase<T>>;

/**
 * Is S T
 * @tparam T T
 * @tparam S S
 */
template <typename T, typename S>
concept IsT = std::is_same_v<T, std::decay_t<S>>;

/**
 * Is P acceptable unary
 * @tparam T T
 * @tparam P P
 */
template <typename T, typename P>
concept IsAcceptableUnary = IsBase<T, P>;

/**
 * Is P Q acceptable binary operation
 * @tparam T T
 * @tparam P P
 * @tparam Q Q
 */
template <typename T, typename P, typename Q>
concept IsAcceptableBinaryEvaluate =
    (IsBase<T, P> || IsT<T, P> || IsBase<T, Q> || IsT<T, Q>)&&(!(IsT<T, P> && IsT<T, Q>));

/**
 * Is P Q acceptable binary assignment
 * @tparam T T
 * @tparam P P
 * @tparam Q Q
 */
template <typename T, typename P, typename Q>
concept IsAcceptableBinaryAssignment = (IsBase<T, P> && (IsT<T, Q> || IsBase<T, Q>));

/**
 * Is P Q purely based
 * @tparam T T
 * @tparam P P
 * @tparam Q Q
 */
template <typename T, typename P, typename Q>
concept IsPureBinary = IsBase<T, P> && IsBase<T, Q>;

/**
 * Some useful helper functions for computation
 * @tparam T T
 * @tparam BaseType Type of base
 */
template <typename T, typename BaseType>
struct ComputeBase {
private:
    using Shape = tensorlib::base::Shape;
    using Index = tensorlib::base::Index;
    using Base = BaseType;

public:
    /**
     * return if b is a subset equal to a in dimension, like, a = [k1, k2, x1, x2], b = [x1, x2]
     * @param a a
     * @param b b
     * @return subseteq or not
     */
    static bool isDimensionBinaryOperable(const Shape &a, const Shape &b) noexcept {

        if (a.size() < b.size())
            return false;

        SizeType m = a.size(), n = b.size(), off = m - n;
        for (SizeType i = 0; i < n; i++) {
            if (a[off + i] != b[i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * throw exception if b is not subseteq to a
     * @param a a
     * @param b b
     */
    static void requireDimensionBinaryOperable(const Shape &a, const Shape &b) {
        if (!isDimensionBinaryOperable(a, b)) {
            throw std::out_of_range(util::functional::messageOf(base::shapeToString(a), " ", base::shapeToString(b),
                                                                "is not subset equal to another"));
        }
    }

    /**
     * Check if a and b are able to perform matrix multiplication
     * @param a a
     * @param b b
     * @return performable or not
     */
    static bool isMatrixMultiplicationOperable(const Shape &a, const Shape &b) noexcept {
        if (a.size() < 2 || b.size() < 2) {
            return false;
        }
        if constexpr (SHAPE_CHECK) {
            SizeType t = std::min(a.size(), b.size());
            for (SizeType i = 0; i < t - 2; i++) {
                if (a[a.size() - t + i] != b[b.size() - t + i]) {
                    return false;
                }
            }
        }
        if (a[a.size() - 1] == b[b.size() - 2]) {
            return true;
        }
        return false;
    }

    /**
     * Get contiguous information of shape in largest dim
     * @tparam Args Arg type, all shape
     * @param shps shapes
     * @return {location contiguous on, size of contiguous}
     */
    template <typename... Args>
        requires(std::is_same_v<std::decay_t<Args>, Shape> && ...)
    static std::pair<SizeType, SizeType> getContiguousInfo(const Shape &dim, Args &&...shps) {
        SizeType contLoc = dim.size() - std::min({shps.size()...});
        SizeType sz = 1;
        for (SizeType i = contLoc, n = dim.size(); i < n; i++) {
            sz *= dim[i];
        }
        return {contLoc, sz};
    }

    /**
     * Recursively decay an index to a specific location
     * @param dimOn dimension on right now
     * @param dimSuff suffix of persuing dimension
     * @param dimTotal total number of dimensions
     * @param finalShape shape of final dimensions
     * @param idxOn index on now
     * @param f function after decayed to call
     * @param args: {index, {totalSize, suffixSize}}
     */
    template <typename... Args>
        requires(std::is_same_v<Args, std::pair<Index *, std::pair<SizeType, SizeType>>> && ...)
    static void recursivelyDecayIndexTo(SizeType dimOn, SizeType dimSuff, SizeType dimTotal, const Shape &finalShape,
                                        Index &idxOn, auto &&f, Args &&...args) {
        SizeType dimLeft = dimTotal - dimOn;
        if (dimLeft <= dimSuff) {
            f(idxOn, (*(args.first))...);
            return;
        }

        for (SizeType i = 0, n = finalShape[dimOn]; i < n; i++) {
            (
                [dimLeft, i](auto &&idx) {
                    if (idx.second.first >= dimLeft && dimLeft > idx.second.second) {
                        (*idx.first)[idx.second.first - dimLeft] = i;
                    }
                }(std::forward<Args>(args)),
                ...);

            idxOn[dimOn] = i;

            recursivelyDecayIndexTo(dimOn + 1, dimSuff, dimTotal, finalShape, idxOn, f, std::forward<Args>(args)...);
        }
    }

    /**
     * Recursively operate
     * @param dimOn dimension on
     * @param dimTotal total dimension
     * @param to base perform on
     * @param idxOn index on
     * @param finalShape final shape
     * @param f function operate
     * @param args {base, index}
     */
    template <typename... Args>
        requires(std::is_same_v<Args, std::pair<Base *, Index *>> && ...)
    static void recursivelyOperate(SizeType dimOn, SizeType dimTotal, Base &to, Index &idxOn, const Shape &finalShape,
                                   auto &&f, Args &&...args) {
        if (dimOn == dimTotal) {
            f(to.get(idxOn), (args.first->get(*(args.second)))...);
            return;
        }

        for (SizeType i = 0, n = finalShape[dimOn]; i < n; i++) {
            (
                [dimOn, i, dimTotal](auto &&pr) {
                    if (dimOn >= dimTotal - pr.first->dimensionSize()) {
                        (*pr.second)[dimOn - (dimTotal - pr.first->dimensionSize())] = i;
                    }
                }(std::forward<Args>(args)),
                ...);

            idxOn[dimOn] = i;

            recursivelyOperate(dimOn + 1, dimTotal, to, idxOn, finalShape, f, std::forward<Args>(args)...);
        }
    }

    /**
     * Operate on contiguous pointers recursively
     * @param contiguousSize size of contiguous
     * @param dimOn dimension on
     * @param dimTo dimension to
     * @param dimTotal dimension total
     * @param to base to
     * @param idxOn index
     * @param finalShape final shape
     * @param f function
     * @param args {base, index}
     */
    template <typename... Args>
        requires(std::is_same_v<Args, std::pair<Base *, Index *>> && ...)
    static void contiguousRecursivelyOperate(SizeType contiguousSize, SizeType dimOn, SizeType dimTo, SizeType dimTotal,
                                             Base &to, Index &idxOn, const Shape &finalShape, auto &&f,
                                             Args &&...args) {
        if (dimOn == dimTo) {
            tensorlib::util::functional::applyOnIterator(to.offsetBy(idxOn), to.offsetBy(idxOn) + contiguousSize,
                                                         std::forward<decltype(f)>(f),
                                                         (args.first->offsetBy(*args.second))...);
            return;
        }

        for (SizeType i = 0, n = finalShape[dimOn]; i < n; i++) {
            (
                [dimOn, i, dimTotal](auto &&pr) {
                    if (dimOn >= dimTotal - pr.first->dimensionSize()) {
                        (*pr.second)[dimOn - (dimTotal - pr.first->dimensionSize())] = i;
                    }
                }(std::forward<Args>(args)),
                ...);

            idxOn[dimOn] = i;

            contiguousRecursivelyOperate(contiguousSize, dimOn + 1, dimTo, dimTotal, to, idxOn, finalShape,
                                         std::forward<decltype(f)>(f), std::forward<Args>(args)...);
        }
    }

    /**
     * A wrapper for recursivelyOperate to change operate into set
     */
    template <typename... Args>
        requires(std::is_same_v<Args, std::pair<Base *, Index *>> && ...)
    static void recursivelySet(SizeType dimOn, SizeType dimTotal, Base &to, Index &idxOn, const Shape &finalShape,
                               auto &&f, Args &&...args) {
        return recursivelyOperate(
            dimOn, dimTotal, to, idxOn, finalShape,
            [&f](auto &x, auto &&...y) { x = f(std::forward<decltype(y)>(y)...); }, std::forward<Args>(args)...);
    }

    /**
     * A wrapper for contiguousRecursivelyOperate to change operate into set
     */
    template <typename... Args>
        requires(std::is_same_v<Args, std::pair<Base *, Index *>> && ...)
    static void contiguousRecursivelySet(SizeType contiguousSize, SizeType dimOn, SizeType dimTo, SizeType dimTotal,
                                         Base &to, Index &idxOn, const Shape &finalShape, auto &&f, Args &&...args) {
        return contiguousRecursivelyOperate(
            contiguousSize, dimOn, dimTo, dimTotal, to, idxOn, finalShape,
            [&f](auto &x, auto &&...y) { x = f(std::forward<decltype(y)>(y)...); }, std::forward<Args>(args)...);
    }

    /**
     * Perform unary operation f on a
     * @param a a
     * @param f f
     * @return f(a)
     */
    static Base unaryEvaluateOperation(Base &a, auto &&f) {
        Base b = Base::ofShape(a.size());

        tensorlib::util::functional::transformOnIterable(b, f, a);

        return b;
    }

    /**
     * Perform unary assignment operation f on a, a := f(a)
     * @param a a
     * @param f f
     */
    static void unaryFunctionAssignOperation(Base &a, auto &&f) {
        tensorlib::util::functional::applyOnIterable(a, f);
    }

    // ============= BINARY EVALUATE =============

    /**
     * Semi binary operation, one side is value
     * @param a a
     * @param t t
     * @param f f
     * @return f(a, t)
     */
    static Base semiBinaryEvaluateOperation(Base &a, const T &t, auto &&f) {
        Base b = Base::ofShape(a.size());

        tensorlib::util::functional::transformOnIterable(b, [&t, &f](const auto &x) { return f(x, t); }, a);

        return b;
    }

    /**
     * Semi binary operation, one side is value
     * @param t t
     * @param a a
     * @param f f
     * @return f(t, a)
     */
    static Base semiBinaryEvaluateOperation(const T &t, Base &a, auto &&f) {
        Base b = Base::ofShape(a.size());

        tensorlib::util::functional::transformOnIterable(b, [&t, &f](const auto &x) { return f(t, x); }, a);

        return b;
    }

    /**
     * Binary evaluate operation
     * @tparam commutative commutative or not, if so, will swap dimension if b's dim is not subseteq to a
     * @param a a
     * @param b b
     * @param f f
     * @return f(a, b)
     */
    template <bool commutative = false>
    static Base binaryEvaluateOperation(Base &a, Base &b, auto &&f) {
        if constexpr (commutative) {
            if (a.dimensionSize() < b.dimensionSize()) {
                return binaryEvaluateOperation(b, a, f);
            }
        }

        auto &shapeA = a.size(), &shapeB = b.size();

        if constexpr (SHAPE_CHECK) {
            requireDimensionBinaryOperable(shapeA, shapeB);
        }

        auto &finalShape = shapeA;
        Base result = Base::ofShape(finalShape);

        auto aIndex = Index(a.dimensionSize(), 0), bIndex = Index(b.dimensionSize(), 0);

        if (a.isContiguous() && b.isContiguous()) {
            auto [dimTo, contSz] = getContiguousInfo(finalShape, shapeA, shapeB);
            contiguousRecursivelySet(contSz, 0, dimTo, finalShape.size(), result, aIndex, finalShape,
                                     std::forward<decltype(f)>(f), std::pair<Base *, Index *>{&a, &aIndex},
                                     std::pair<Base *, Index *>{&b, &bIndex});
        } else {
            recursivelySet(0, finalShape.size(), result, aIndex, finalShape, std::forward<decltype(f)>(f),
                           std::pair<Base *, Index *>{&a, &aIndex}, std::pair<Base *, Index *>{&b, &bIndex});
        }

        return result;
    }

    /**
     * Distributor for binary
     * @tparam commutative commutative or not
     * @param a a
     * @param b b
     * @param f f
     * @return f(a, b)
     */
    template <typename A, typename B, bool commutative = false>
        requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
    static Base binaryEvaluateDistributor(A &&a, B &&b, auto &&f) {
        if constexpr (computational_cpu::IsPureBinary<T, A, B>) {
            return binaryEvaluateOperation<commutative>(a, b, std::forward<decltype(f)>(f));
        } else {
            return semiBinaryEvaluateOperation(a, b, std::forward<decltype(f)>(f));
        }
    }

    // ============= BINARY FUNCTION ASSIGNMENT =============

    /**
     * Binary assignment operation to = f(a)
     * @param to to
     * @param a a
     * @param f f
     */
    static void binaryFunctionAssignOperation(Base &to, Base &a, auto &&f) {
        auto &shapeTo = to.size(), &shapeA = a.size();

        if constexpr (SHAPE_CHECK) {
            requireDimensionBinaryOperable(shapeTo, shapeA);
        }

        auto toIndex = Index(to.dimensionSize(), 0), aIndex = Index(a.dimensionSize(), 0);

        if (to.isContiguous() && a.isContiguous()) {
            auto [dimTo, contSz] = getContiguousInfo(shapeTo, shapeA);
            contiguousRecursivelyOperate(contSz, 0, dimTo, shapeTo.size(), to, aIndex, shapeTo,
                                         std::forward<decltype(f)>(f), std::pair<Base *, Index *>{&a, &aIndex});
        } else {
            recursivelyOperate(0, toIndex.size(), to, toIndex, shapeTo, f, std::pair<Base *, Index *>{&a, &aIndex});
        }
    }

    /**
     * Semibinary assignment operation, to = f(t)
     * @param to to
     * @param t t
     * @param f f
     */
    static void semiBinaryFunctionAssignOperation(Base &to, const T &t, auto &&f) {
        tensorlib::util::functional::applyOnIterable(to, [&f, &t](auto &x) { f(x, t); });
    }

    /**
     * a = f(b)
     * @param a a
     * @param b b
     * @param f f
     */
    template <typename A, typename B, typename F>
        requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
    static void binaryAssignmentDistributor(A &&a, B &&b, F &&f) {
        if constexpr (computational_cpu::IsPureBinary<T, A, B>) {
            binaryFunctionAssignOperation(a, b, std::forward<F>(f));
        } else {
            semiBinaryFunctionAssignOperation(a, b, std::forward<F>(f));
        }
    }

    // ============= SUFFIX DIMENSIONAL EVALUATE =============

    /**
     * Using recursivelyDecayIndexTo to perform unary decay
     * @param toShape shape of result
     * @param toSuff result's suffix dimension left
     * @param aDim a's dimension
     * @param aSuff a's suffix dimension left
     * @param f f
     * @param toRemain perserved index for result
     * @param aRemain perserved index for a
     */
    template <typename F>
    static void suffixUnaryIteration(const Shape &toShape, SizeType toSuff, SizeType aDim, SizeType aSuff, F &&f,
                                     SizeType toRemain = 0, SizeType aRemain = 0) {
        Index idxOn = Index(toShape.size() - toSuff + toRemain);
        Index idxA = Index(aDim - aSuff + aRemain);
        recursivelyDecayIndexTo(0, toSuff, toShape.size(), toShape, idxOn, std::forward<F>(f),
                                std::pair<Index *, std::pair<SizeType, SizeType>>{&idxA, {aDim, aSuff}});
    }

    /**
     * Using recursivelyDecayIndexTo to perform binary decay
     * @param toShape shape of result
     * @param toSuff result's suffix dimension left
     * @param aDim a's dimension
     * @param aSuff a's suffix dimension left
     * @param bDim b's dimension
     * @param bSuff b's suffix
     * @param f f
     * @param toRemain perserved index for result
     * @param aRemain perserved index for a
     * @param bRemain perserved index for b
     */
    template <typename F>
    static void suffixBinaryIteration(const Shape &toShape, SizeType toSuff, SizeType aDim, SizeType aSuff,
                                      SizeType bDim, SizeType bSuff, F &&f, SizeType toRemain = 0, SizeType aRemain = 0,
                                      SizeType bRemain = 0) {
        Index idxOn = Index(toShape.size() - toSuff + toRemain);
        Index idxA = Index(aDim - aSuff + aRemain);
        Index idxB = Index(bDim - bSuff + bRemain);
        recursivelyDecayIndexTo(0, toSuff, toShape.size(), toShape, idxOn, std::forward<F>(f),
                                std::pair<Index *, std::pair<SizeType, SizeType>>{&idxA, {aDim, aSuff}},
                                std::pair<Index *, std::pair<SizeType, SizeType>>{&idxB, {bDim, bSuff}});
    }

    // =============== HELPER FUNCTIONS ===============

    /**
     * Matrix multiplcation
     * @param a A
     * @param b B
     * @return AB
     */
    template <typename A, typename B>
        requires computational_cpu::IsPureBinary<T, A, B>
    static Base matrixMultiplyMatrix(A &&a, B &&b) {
        auto &aShape = a.size(), &bShape = b.size();
        auto n = std::max(aShape.size(), bShape.size());
        Shape toShape(n);
        if (aShape.size() >= bShape.size()) {
            for (SizeType i = 0; i < n - 2; i++)
                toShape[i] = aShape[i];
        } else {
            for (SizeType i = 0; i < n - 2; i++)
                toShape[i] = bShape[i];
        }
        toShape[n - 2] = aShape[aShape.size() - 2];
        toShape.back() = bShape.back();

        Base to = Base::ofShape(toShape);

        suffixBinaryIteration(
            toShape, 2, aShape.size(), 2, bShape.size(), 2,
            [&to, &a, &b, n = toShape.size(), ai = a.dimensionSize() - 2, aj = a.dimensionSize() - 1,
             bi = b.dimensionSize() - 2, bj = b.dimensionSize() - 1, p = aShape[aShape.size() - 2], q = aShape.back(),
             r = bShape.back()](auto &&toIdx, auto &&aIdx, auto &&bIdx) mutable {
                for (SizeType i = 0; i < p; i++) {
                    toIdx.back() = i;
                    aIdx.back() = i;
                    auto toi = to[toIdx];
                    auto aTmp = a[aIdx];
                    for (SizeType j = 0; j < q; j++) {
                        bIdx.back() = j;
                        auto bTmp = b[bIdx];
                        auto &va = aTmp.get({j});
                        for (SizeType k = 0; k < r; k++) {
                            toi.get({k}) += va * bTmp.get({k});
                        }
                    }
                }
            },
            1, 1, 1);

        return to;
    }

    /**
     * Wrapper of matrix mul, in case other functions will needed to be implemented in future
     * @param a A
     * @param b B
     * @return AB
     */
    template <typename A, typename B>
    static Base matrixMultiplyDistributor(A &&a, B &&b) {
        return matrixMultiplyMatrix(std::forward<A>(a), std::forward<B>(b));
    }

    /**
     * Conv2D
     * @param a a
     * @param b b
     * @return a conv2d b
     */
    template <typename A, typename B>
        requires computational_cpu::IsPureBinary<T, A, B>
    static Base conv2d(A &&a, B &&b) {

        static int cnt = 0;
        cnt++;
        // a = [..., C, H, W]
        // b = [D, C, X, Y]
        // a * b = [..., D, H - X + 1, W - X + 1]

        auto finalShape = a.size();
        auto &bShape = b.size();
        SizeType C = finalShape[finalShape.size() - 3], D = bShape[0];

        finalShape[finalShape.size() - 3] = bShape[0];
        finalShape[finalShape.size() - 2] -= bShape[2] - 1;
        finalShape[finalShape.size() - 1] -= bShape[3] - 1;

        SizeType H = finalShape[finalShape.size() - 2];
        SizeType W = finalShape[finalShape.size() - 1];
        SizeType X = bShape[2];
        SizeType Y = bShape[3];

        Base result = Base::ofShape(finalShape);

        if (a.isContiguous() && b.isContiguous()) {

            // [B, C, aH, aW]
            T *pa = a.raw();
            // [D, C, X, Y]
            T *pker = b.raw();
            // [B, D, H, W]
            T *pres = result.raw();
            SizeType bnum = finalShape[0];

            SizeType aH = H + X - 1, aW = W + Y - 1;
            for (SizeType d = 0; d < D; d++) {
                for (SizeType c = 0; c < C; c++) {
                    // pker: [X, Y]
                    for (SizeType batch = 0; batch < bnum; batch++) {
                        // pabatch: [H * W]
                        T *pabatch = pa + batch * C * aH * aW + c * aH * aW;
                        // presbatch: [oH * oW]
                        T *presbatch = pres + batch * D * H * W + d * H * W;
                        for (SizeType i = 0; i < H; i++) {
                            for (SizeType j = 0; j < W; j++) {
                                auto &v = presbatch[i * W + j];
                                for (SizeType x = 0; x < X; x++) {
                                    for (SizeType y = 0; y < Y; y++) {
                                        v += pker[x * Y + y] * pabatch[(i + x) * aW + (j + y)];
                                    }
                                }
                            }
                        }
                    }
                    pker += X * Y;
                }
            }
        } else {
            suffixBinaryIteration(
                finalShape, 3, a.dimensionSize(), 3, 4, 3,
                [&result, &a, &b, loc = finalShape.size() - 3, C, D, H, W, X, Y](auto &&toIdx, auto &&aIndex,
                                                                                 auto &&bIndex) mutable {
                    for (SizeType d = 0; d < D; d++) {
                        toIdx[toIdx.size() - 3] = d;
                        bIndex[0] = d;
                        for (SizeType i = 0; i < H; i++) {
                            toIdx[toIdx.size() - 2] = i;
                            for (SizeType j = 0; j < W; j++) {
                                toIdx[toIdx.size() - 1] = j;
                                bIndex[0] = d;

                                auto &v = result.get(toIdx);
                                v = ComputationalConstant<T>::Zero;

                                for (SizeType c = 0; c < C; c++) {
                                    bIndex[1] = c;
                                    aIndex[loc] = c;
                                    for (SizeType x = 0; x < X; x++) {
                                        bIndex[2] = x;
                                        aIndex[loc + 1] = i + x;
                                        for (SizeType y = 0; y < Y; y++) {
                                            bIndex[3] = y;
                                            aIndex[loc + 2] = j + y;
                                            v += a.get(aIndex) * b.get(bIndex);
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                3, 3, 3);
        }

        return result;
    }

    /**
     * Padder a by dim
     * @param a a
     * @param d dim
     * @return a pad dim
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base pad(A &&a, auto &&d) {
        auto &aShape = a.size();
        Shape toShape = aShape;
        SizeType n = toShape.size();
        for (SizeType i = 0; i < n; i++) {
            toShape[i] += d[i].first + d[i].second;
        }

        Base result = Base::ofShape(toShape);

        Index idx = Index(toShape.size());
        Index toIdx = Index(toShape.size());

        recursivelyDecayIndexTo(0, 0, n, aShape, idx, [&result, &a, &d, n, &toIdx](auto &&aIdx) {
            for (SizeType i = 0; i < n; i++) {
                toIdx[i] = aIdx[i] + d[i].first;
            }
            result.get(toIdx) = a.get(aIdx);
        });

        return result;
    }

    /**
     * Cut a on dim
     * @param a a
     * @param d d
     * @return a cut on dim
     */
    template <typename A>
        requires computational_cpu::IsBase<T, A>
    static Base cut(A &&a, auto &&d) {
        auto &aShape = a.size();
        SizeType n = aShape.size();
        Shape toShape = Shape(n);
        for (SizeType i = 0; i < n; i++) {
            toShape[i] = d[i].second - d[i].first;
        }

        return Base(a, std::move(toShape), [n, d](const auto &idx) {
            auto nidx = idx;
            for (SizeType i = 0; i < n; i++) {
                nidx[i] += d[i].first;
            }
            return nidx;
        });
    }

    //    template <typename A>
    //        requires computational_cpu::IsBase<T, A>
    //    static Base maxBack(A &&a) {
    //        auto &aShape = a.size();
    //        SizeType n = aShape.size();
    //        Shape toShape = Shape(n - 1);
    //        for (SizeType i = 0; i < n - 1; i++) {
    //            toShape[i] = aShape[i];
    //        }
    //
    //        Base result = Base::ofShape(toShape);
    //
    //        if (a.isContiguous()) {
    //            suffixUnaryIteration(
    //                toShape, 0, n, 1,
    //                [&result, &a, m = aShape.back()](auto &&idx, auto &&aIdx) {
    //                    auto &t = result.get(idx);
    //                    aIdx.back() = 0;
    //                    auto arr = a.offsetBy(aIdx);
    //                    for (SizeType i = 0; i < m; i++) {
    //                        t = std::max(t, arr[i]);
    //                    }
    //                },
    //                0, 1);
    //        } else {
    //            suffixUnaryIteration(
    //                toShape, 0, n, 1,
    //                [&result, &a, m = aShape.back()](auto &&idx, auto &&aIdx) {
    //                    auto &t = result.get(idx);
    //                    for (SizeType i = 0; i < m; i++) {
    //                        aIdx.back() = i;
    //                        t = std::max(t, a.get(aIdx));
    //                    }
    //                },
    //                0, 1);
    //        }
    //
    //        return result;
    //    }
    //
    //    template <typename A>
    //        requires computational_cpu::IsBase<T, A>
    //    static Base maxBackMask(A &&a) {
    //        auto &aShape = a.size();
    //        SizeType n = aShape.size();
    //
    //        Base result = Base::ofShape(aShape);
    //
    //        if (a.isContiguous()) {
    //            suffixUnaryIteration(
    //                aShape, 1, n, 1,
    //                [&result, &a, m = aShape.back()](auto &&idx, auto &&aIdx) {
    //                    auto tarr = result.offsetBy(idx);
    //                    aIdx.back() = 0;
    //                    auto aarr = a.offsetBy(aIdx);
    //                    T mx = aarr[0];
    //                    SizeType mxi = 0;
    //                    for (SizeType i = 1; i < m; i++) {
    //                        if (aarr[i] > mx) {
    //                            mxi = i;
    //                            mx = aarr[i];
    //                        }
    //                    }
    //                    tarr[mxi] = ComputationConstant<T>::One;
    //                },
    //                1, 1);
    //        } else {
    //            suffixUnaryIteration(
    //                aShape, 1, n, 1,
    //                [&result, &a, m = aShape.back()](auto &&idx, auto &&aIdx) {
    //                    auto tarr = result.offsetBy(idx);
    //                    aIdx.back() = 0;
    //                    T mx = a.get(aIdx);
    //                    SizeType mxi = 0;
    //                    for (SizeType i = 1; i < m; i++) {
    //                        T v;
    //                        if ((v = a.get(aIdx)) > mx) {
    //                            mxi = i;
    //                            mx = v;
    //                        }
    //                    }
    //                    tarr[mxi] = ComputationConstant<T>::One;
    //                },
    //                1, 1);
    //        }
    //
    //        return result;
    //    }

    // ============= POINTER LEVEL OPERATIONS ==============
    //      WE ASSUME ALL OF THOSE BASE ARE CONTIGUOUS
    struct PointerLevel {

        /**
         * Recursively operate on pointer level
         */
        template <typename... Args>
            requires(std::is_same_v<Args, std::pair<Base *, Index *>> && ...)
        static void recursivelyOperate(SizeType contiguousSize, SizeType dimOn, SizeType dimTo, SizeType dimTotal,
                                       Base &to, Index &idxOn, const Shape &finalShape, auto &&f, Args &&...args) {
            if (dimOn == dimTo) {
                f(contiguousSize, to.offsetBy(idxOn), (args.first->offsetBy(*args.second))...);
                return;
            }

            for (SizeType i = 0, n = finalShape[dimOn]; i < n; i++) {
                (
                    [dimOn, i, dimTotal](auto &&pr) {
                        if (dimOn >= dimTotal - pr.first->dimensionSize()) {
                            (*pr.second)[dimOn - (dimTotal - pr.first->dimensionSize())] = i;
                        }
                    }(std::forward<Args>(args)),
                    ...);

                idxOn[dimOn] = i;

                PointerLevel::recursivelyOperate(contiguousSize, dimOn + 1, dimTo, dimTotal, to, idxOn, finalShape,
                                                 std::forward<decltype(f)>(f), std::forward<Args>(args)...);
            }
        }

        /**
         * Perform multinary operation on pointer level
         * @param N Total size
         * @param contiguousSize contiguous size
         * @param f function
         * @param args {ptr, {jump, index now, total size}}
         */
        template <typename... Args>
            requires(std::is_same_v<std::decay_t<Args>, std::pair<T *, std::tuple<SizeType, SizeType, SizeType>>> &&
                     ...)
        static void multinaryOperation(SizeType N, SizeType contiguousSize, auto &&f, Args &&...args) {
            ((std::get<0>(args.second) = std::get<0>(args.second) == 0 ? contiguousSize : std::get<0>(args.second)),
             ...);
            for (SizeType i = 0; i < N; i += contiguousSize) {
                f(contiguousSize, (args.first + std::get<1>(args.second))...);
                (((std::get<1>(args.second) += std::get<0>(args.second)) %= std::get<2>(args.second)), ...);
            }
        }

        /**
         * Perform pure binary evaluate operation
         * @tparam Commutative commutative or not
         * @param a a
         * @param b b
         * @param f f
         * @return f(a, b)
         */
        template <bool Commutative>
        static Base pureBinaryEvaluateOperation(Base &a, Base &b, auto &&f) {
            if constexpr (Commutative) {
                if (a.dimensionSize() < b.dimensionSize()) {
                    return binaryEvaluateOperation(b, a, f);
                }
            }

            if constexpr (SHAPE_CHECK) {
                auto &shapeA = a.size(), &shapeB = b.size();
                requireDimensionBinaryOperable(shapeA, shapeB);
            }

            Base result = Base::ofShape(a.size());
            auto ptr = result.raw(), aptr = a.raw(), bptr = b.raw();

            for (SizeType idx = 0, n = a.getTotalSize(), jmp = b.getTotalSize(); idx < n; idx += jmp) {
                f(jmp, ptr + idx, aptr + idx, bptr);
            }

            return result;
        }

        /**
         * Same as non pointer level
         */
        static Base semiBinaryEvaluateOperation(Base &a, const T &b, auto &&f) {
            Base result = Base::ofShape(a.size());
            f(result.getTotalSize(), result.raw(), a.raw(), b);
            return result;
        }

        /**
         * Same as non pointer level
         */
        static Base semiBinaryEvaluateOperation(const T &a, Base &b, auto &&f) {
            Base result = Base::ofShape(b.size());
            f(result.getTotalSize(), result.raw(), a, b.raw());
            return result;
        }

        /**
         * Similar to the distributor
         * @tparam Commutative commutative or not
         * @param a a
         * @param b b
         * @param f1 used for pure binary
         * @param f2 used for a base, b T
         * @param f3 used for a T, b base
         * @return f(a, b)
         */
        template <typename A, typename B, bool Commutative = false>
            requires computational_cpu::IsAcceptableBinaryEvaluate<T, A, B>
        static Base binaryEvaluateOperation(A &&a, B &&b, auto &&f1, auto &&f2, auto &&f3) {
            if constexpr (computational_cpu::IsPureBinary<T, A, B>) {
                return PointerLevel::pureBinaryEvaluateOperation<Commutative>(a, b, f1);
            } else if constexpr (computational_cpu::IsT<T, B>) {
                return PointerLevel::semiBinaryEvaluateOperation(a, b, f2);
            } else {
                return PointerLevel::semiBinaryEvaluateOperation(a, b, f3);
            }
        }

        /**
         * Same as non pointer level
         */
        static void binaryFunctionAssignOperation(Base &to, Base &a, auto &&f) {
            if constexpr (SHAPE_CHECK) {
                auto &shapeTo = to.size(), &shapeA = a.size();
                requireDimensionBinaryOperable(shapeTo, shapeA);
            }

            auto toptr = to.raw(), aptr = a.raw();

            for (SizeType idx = 0, n = to.getTotalSize(), stp = a.getTotalSize(); idx < n; idx += stp) {
                f(stp, toptr + idx, aptr);
            }
        }

        /**
         * Same as non pointer level
         */
        static void semiBinaryFunctionAssignOperation(Base &to, const T &t, auto &&f) {
            f(to.getTotalSize(), to.raw(), t);
        }

        /**
         * Similar to non pointer level
         * @param a a
         * @param b b
         * @param f1 used for pure binary
         * @param f2 used for semi binary
         */
        template <typename A, typename B>
            requires computational_cpu::IsAcceptableBinaryAssignment<T, A, B>
        static void binaryAssignmentOperation(A &&a, B &&b, auto &&f1, auto &&f2) {
            if constexpr (computational_cpu::IsBase<T, B>) {
                PointerLevel::binaryFunctionAssignOperation(a, b, std::forward<decltype(f1)>(f1));
            } else {
                PointerLevel::semiBinaryFunctionAssignOperation(a, b, std::forward<decltype(f2)>(f2));
            }
        }

        /**
         * Same as non pointer level
         */
        template <typename A>
            requires computational_cpu::IsBase<T, A>
        static Base unaryOperation(A &&a, auto &&f) {
            Base result = Base::ofShape(a.size());
            f(result.getTotalSize(), result.raw(), a.raw());
            return result;
        }
    };

    // ============= SPECIFIC DIMENSIONAL =============

    /**
     * Get bool array that if dimension is appear in dim in shp
     * @param shp shp
     * @param dim dim
     * @return appear or not
     */
    static std::unique_ptr<bool[]> getDimensionArray(const Shape &shp,
                                                     const Index &dim) noexcept(!SHAPE_CHECK && NEW_NOEXCEPT) {
        SizeType n = shp.size();

        std::unique_ptr<bool[]> result(new bool[n]());

        for (auto x : dim) {
            if constexpr (SHAPE_CHECK) {
                if (x >= n)
                    throw std::out_of_range(std::string{"Dimension "} + std::to_string(x) + " greater than size " +
                                            std::to_string(n));
            }
            result[x] = true;
        }

        return result;
    }

    /**
     * Perform operation on a specific dimension, only set idx if is choosed dim
     * @param dimOn dimension on
     * @param dimCnt dimension count
     * @param to result
     * @param idx index
     * @param dimChoice choice of dimension
     * @param a a
     * @param aShape a's shape
     * @param aIndex a's index
     * @param f function
     */
    static void performOperationOnDimension(SizeType dimOn, SizeType dimCnt, Base &to, Index &idx,
                                            const std::unique_ptr<bool[]> &dimChoice, Base &a, const Shape &aShape,
                                            Index &aIndex, auto &&f) {
        if (dimOn == aIndex.size()) {
            f(to.get(idx), a.get(aIndex));
            return;
        }
        if (dimChoice[dimOn]) {
            for (SizeType i = 0, n = aShape[dimOn]; i < n; i++) {
                aIndex[dimOn] = i;

                performOperationOnDimension(dimOn + 1, dimCnt, to, idx, dimChoice, a, aShape, aIndex, f);
            }
        } else {
            for (SizeType i = 0, n = aShape[dimOn]; i < n; i++) {
                aIndex[dimOn] = i;
                idx[dimCnt] = i;

                performOperationOnDimension(dimOn + 1, dimCnt + 1, to, idx, dimChoice, a, aShape, aIndex, f);
            }
        }
    }

    /**
     * Perform operation on dimension
     * @param a a
     * @param dim dim
     * @param f f
     * @param t t
     * @return f(a on dim)
     */
    template <typename A, typename F>
        requires computational_cpu::IsAcceptableUnary<T, A>
    static Base dimensionalOperation(A &&a, const Index &dim, F &&f, const T &t = T()) {
        auto &aShape = a.size();
        std::unique_ptr<bool[]> dimChoice = getDimensionArray(aShape, dim);

        SizeType m = aShape.size();
        for (SizeType i = 0; i < aShape.size(); i++) {
            if (dimChoice[i]) {
                m--;
            }
        }

        Shape shp(m, 0);
        for (SizeType i = 0, cnt = 0; i < aShape.size(); i++) {
            if (!dimChoice[i]) {
                shp[cnt++] = aShape[i];
            }
        }

        Base b = Base::ofShape(shp, t);
        Index idx = Index(m, 0);

        Index aIndex = Index(aShape.size(), 0);

        performOperationOnDimension(0, 0, b, idx, dimChoice, a, aShape, aIndex, std::forward<F>(f));

        return b;
    }

    /**
     * Fast power
     * @param t
     * @param p
     * @return t^p
     */
    static T fastPow(const T &t, SizeType p) {
        T n(tensorlib::tensor::ComputationConstant<T>::One), b(t);

        for (; p; p >>= 1) {
            if (~p & 1) {
                n *= b;
            }
            b *= b;
        }

        return n;
    }
};
}; // namespace computational_cpu

}; // namespace computational
}; // namespace tensor
}; // namespace tensorlib

#endif