#ifndef __TLIB__TENSOR_OPERATIONS_HPP__
#define __TLIB__TENSOR_OPERATIONS_HPP__

#include "tlib/tensor/tensor.hpp"
#include "tlib/tensor/tensor_print.hpp"

namespace tensorlib {
namespace tensor {

/**
 * Gradienter decider, returns a nullptr if not require gradient, or a gradienter if required
 * @tparam T Type
 * @tparam Device Device
 * @param shape shape of gradient
 * @param f (grad, args...) -> void, used for back prop
 * @param args all arguments. if one of them needs gradient, then return a gradienter
 * @return true
 */
template <typename T, typename Device>
std::shared_ptr<Gradienter<T, Device>> gradientDecider(tensorlib::base::Shape shape, auto &&f, auto &&...args) {
    if (((!args.isRequireGrad()) && ...)) {
        return nullptr;
    } else {
        return make_shared<Gradienter<T, Device>>(
            std::move(shape),
            [f = std::forward<decltype(f)>(f), ... args = std::forward<decltype(args)>(args).getGradienter()](
                BaseTensor<T, Device> &g) mutable { f(g, std::forward<decltype(args)>(args)...); },
            std::vector{args.getGradienter()...});
    }
}

#define TRY_ADD_GRAD(g, v)                                                                                             \
    do {                                                                                                               \
        if ((g) != nullptr) {                                                                                          \
            Gradienter<T, Device>::addGradient((g), (v));                                                              \
        }                                                                                                              \
    } while (0);

namespace operations {

template <typename T, typename Device>
Tensor<T, Device> operator-(const Tensor<T, Device> &a) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::negate(a.getBase()));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [x = a.acquireBase(), shpSize = result->dimensionSize()](auto &&g, auto &&xgrad) mutable {
                        TRY_ADD_GRAD(xgrad, (Computation<T, Device>::negate(g)));
                    },
                    a));
}

template <typename T, typename Device>
Tensor<T, Device> operator+(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::add(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result,
        gradientDecider<T, Device>(
            result->size(),
            [x = a.acquireBase(), y = b.acquireBase(), shpSize = result->dimensionSize()](auto &&g, auto &&xgrad,
                                                                                          auto &&ygrad) mutable {
                TRY_ADD_GRAD(xgrad, (Computation<T, Device>::sum(g, tensorlib::base::Index::generateSequenceInRange(
                                                                        0, shpSize - x->dimensionSize()))));
                TRY_ADD_GRAD(ygrad, (Computation<T, Device>::sum(g, tensorlib::base::Index::generateSequenceInRange(
                                                                        0, shpSize - y->dimensionSize()))));
            },
            a, b));
}

template <typename T, typename Device>
Tensor<T, Device> operator+(const Tensor<T, Device> &a, const T &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::add(a.getBase(), b));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(), [](auto &&g, auto &&xgrad) { TRY_ADD_GRAD(xgrad, (g)); }, a));
}

template <typename T, typename Device>
Tensor<T, Device> operator+(const T &a, const Tensor<T, Device> &b) {
    return b + a;
}

template <typename T, typename Device>
Tensor<T, Device> operator-(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::minus(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [x = a.acquireBase(), y = b.acquireBase(),
                     shpSize = result->dimensionSize()](auto &&g, auto &&xgrad, auto &&ygrad) mutable {
                        TRY_ADD_GRAD(xgrad, g);
                        TRY_ADD_GRAD(
                            ygrad,
                            (Computation<T, Device>::negate(Computation<T, Device>::sum(
                                g, tensorlib::base::Index::generateSequenceInRange(0, shpSize - y->dimensionSize())))));
                    },
                    a, b));
}

template <typename T, typename Device>
Tensor<T, Device> operator-(const Tensor<T, Device> &a, const T &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::minus(a.getBase(), b));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(result->size(), [](auto &&g, auto &&xgrad) { TRY_ADD_GRAD(xgrad, g); }, a));
}

template <typename T, typename Device>
Tensor<T, Device> operator*(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::mul(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result,
        gradientDecider<T, Device>(
            result->size(),
            [x = a.acquireBase(), y = b.acquireBase(), shpSize = result->dimensionSize()](auto &&g, auto &&xgrad,
                                                                                          auto &&ygrad) mutable {
                TRY_ADD_GRAD(xgrad, (Computation<T, Device>::sum(Computation<T, Device>::mul(g, *y),
                                                                 tensorlib::base::Index::generateSequenceInRange(
                                                                     0, shpSize - x->dimensionSize()))));

                TRY_ADD_GRAD(ygrad, (Computation<T, Device>::sum(Computation<T, Device>::mul(g, *x),
                                                                 tensorlib::base::Index::generateSequenceInRange(
                                                                     0, shpSize - y->dimensionSize()))));
            },
            a, b));
}

template <typename T, typename Device>
Tensor<T, Device> operator*(const Tensor<T, Device> &a, const T &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::mul(a.getBase(), b));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [k = b](auto &&g, auto &&xgrad) { TRY_ADD_GRAD(xgrad, (Computation<T, Device>::mul(g, k))); }, a));
}

template <typename T, typename Device>
Tensor<T, Device> operator*(const T &a, const Tensor<T, Device> &b) {
    return b * a;
}

template <typename T, typename Device>
Tensor<T, Device> operator/(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::div(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [x = a.acquireBase(), y = b.acquireBase(),
                     shpSize = result->dimensionSize()](auto &&g, auto &&xgrad, auto &&ygrad) mutable {
                        TRY_ADD_GRAD(xgrad, (Computation<T, Device>::div(g, *y)));

                        TRY_ADD_GRAD(ygrad,
                                     (Computation<T, Device>::negate(Computation<T, Device>::div(
                                         Computation<T, Device>::sum(Computation<T, Device>::mul(g, *x),
                                                                     tensorlib::base::Index::generateSequenceInRange(
                                                                         0, shpSize - y->dimensionSize())),
                                         Computation<T, Device>::mul(*y, *y)))));
                    },
                    a, b));
}

template <typename T, typename Device>
Tensor<T, Device> operator/(const Tensor<T, Device> &a, const T &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::div(a.getBase(), b));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [k = T(b)](auto &&g, auto &&xgrad) {
                        TRY_ADD_GRAD(xgrad, (Computation<T, Device>::mul(g, ComputationConstant<T>::One / k)));
                    },
                    a));
}

template <typename T, typename Device>
Tensor<T, Device> &operator+=(Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    return a = (a + b);
}

template <typename T, typename Device, typename S>
    requires std::is_constructible_v<S, T>

Tensor<T, Device> &operator+=(Tensor<T, Device> &a, const S &b) {
    return a = (a + b);
}

template <typename T, typename Device>
Tensor<T, Device> &operator-=(Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    return a = (a - b);
}

template <typename T, typename Device, typename S>
    requires std::is_constructible_v<S, T>

Tensor<T, Device> &operator-=(Tensor<T, Device> &a, const S &b) {
    return a = (a - b);
}

template <typename T, typename Device>
Tensor<T, Device> &operator*=(Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    return a = (a * b);
}

template <typename T, typename Device, typename S>
    requires std::is_constructible_v<S, T>

Tensor<T, Device> &operator*=(Tensor<T, Device> &a, const S &b) {
    return a = (a * T(b));
}

template <typename T, typename Device>
Tensor<T, Device> &operator/=(Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    return a = (a / b);
}

template <typename T, typename Device, typename S>
    requires std::is_constructible_v<S, T>
Tensor<T, Device> &operator/=(Tensor<T, Device> &a, const S &b) {
    return a = (a / T(b));
}

}; // namespace operations

namespace functional {

/**
 * Calculate sum of all value in tensor
 * @tparam T Type
 * @tparam Device Device
 * @param a tensor
 * @return sum
 */
template <typename T, typename Device>
Tensor<T, Device> sum(const Tensor<T, Device> &a) {
    return Tensor<T, Device>(
        Computation<T, Device>::sum(a.getBase()),
        gradientDecider<T, Device>(
            tensorlib::base::Shape(0),
            [xShape = a.size()](auto &&g, auto &&xgrad) {
                TRY_ADD_GRAD(xgrad, (Computation<T, Device>::mul(
                                        BaseTensor<T, Device>::ofShape(xShape, ComputationConstant<T>::One), g)));
            },
            a));
}

/**
 * Calculate sum along specific dimension of tensor
 * @tparam T Type
 * @tparam Device Device
 * @param a tensor
 * @param dim dimension
 * @return sum
 */
template <typename T, typename Device>
Tensor<T, Device> sum(const Tensor<T, Device> &a, const tensorlib::base::Index &dim) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::sum(a.getBase(), dim));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [expandDim =
                                              [&a, &dim]() {
                                                  auto shp = a.size();
                                                  std::vector<std::pair<SizeType, SizeType>> dm;
                                                  for (auto x : dim) {
                                                      dm.push_back({x, shp[x]});
                                                  }
                                                  return dm;
                                              }()](auto &&g, auto &&xgrad) {
                                             TRY_ADD_GRAD(xgrad, (Computation<T, Device>::expand(g, expandDim)));
                                         },
                                         a));
}

/**
 * Calculate exp of all values in tensor
 * @tparam T Type
 * @tparam Device Device
 * @param x x
 * @return exp(x)
 */
template <typename T, typename Device>
auto exp(const Tensor<T, Device> &x) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::exp(x.getBase()));
    return Tensor<T, Device>(result,
                             gradientDecider<T, Device>(
                                 result->size(),
                                 [result](auto &&g, auto &&xgrad) {
                                     Gradienter<T, Device>::addGradient(xgrad, Computation<T, Device>::mul(g, *result));
                                 },
                                 x));
}

/**
 * Calculate log of all values in tensor
 * @tparam T Type
 * @tparam Device Deice
 * @param x x
 * @return log(x)
 */
template <typename T, typename Device>
auto log(const Tensor<T, Device> &x) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::log(x.getBase()));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [p = x.acquireBase()](auto &&g, auto &&xgrad) {
                                             TRY_ADD_GRAD(xgrad, (Computation<T, Device>::div(g, *p)));
                                         },
                                         x));
}

/**
 * Calculate absolute value of all tensors in tensor
 * @tparam T Type
 * @tparam Device Device
 * @param x x
 * @return |x|
 */
template <typename T, typename Device>
auto abs(const Tensor<T, Device> &x) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::abs(x.getBase()));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [orig = x.acquireBase()](auto &&g, auto &&xgrad) {
                                             TRY_ADD_GRAD(xgrad, (Computation<T, Device>::apply(*orig, [](auto &&p) {
                                                              return p >= ComputationConstant<T>::Zero
                                                                         ? ComputationConstant<T>::One
                                                                         : -ComputationConstant<T>::One;
                                                          })));
                                         },
                                         x));
}

/**
 * Calculate max value of x, y
 * @tparam T Type
 * @tparam Device Device
 * @param x x
 * @param y y
 * @return max(x, y)
 */
template <typename T, typename Device>
auto max(const Tensor<T, Device> &x, const Tensor<T, Device> &y) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::max(x.getBase(), y.getBase()));
    std::shared_ptr<BaseTensor<T, Device>> xMsk = nullptr, yMsk = nullptr;

    if (x.isRequireGrad()) {
        xMsk = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::maxMask(x.getBase(), y.getBase()));
    }
    if (y.isRequireGrad()) {
        yMsk = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::maxMask(y.getBase(), x.getBase()));
    }

    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [xMsk = std::move(xMsk), yMsk = std::move(yMsk)](auto &&g, auto &&xgrad, auto &&ygrad) mutable {
                        TRY_ADD_GRAD(xgrad, (Computation<T, Device>::mul(g, *xMsk)));
                        TRY_ADD_GRAD(ygrad, (Computation<T, Device>::mul(g, *yMsk)));
                    },
                    x, y));
}

/**
 * Matrix multiply a, b
 * @tparam T Type
 * @tparam Device Device
 * @param a A
 * @param b B
 * @return AB
 */
template <typename T, typename Device>
Tensor<T, Device> matMul(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::matMul(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result,
        gradientDecider<T, Device>(
            result->size(),
            [xShape = a.size(), yShape = b.size(),
             xSum = tensorlib::base::Index::generateIncreasingSequence(result->dimensionSize() -
                                                                       a.getBase().dimensionSize()),
             ySum = tensorlib::base::Index::generateIncreasingSequence(result->dimensionSize() -
                                                                       b.getBase().dimensionSize()),
             x = a.acquireBase(), y = b.acquireBase()](auto &&g, auto &&xgrad, auto &&ygrad) mutable {
                using Compute = Computation<T, Device>;

                if (xgrad != nullptr) {
                    auto xDiff = Compute::matMul(g, Compute::transpose(*y));
                    Gradienter<T, Device>::addGradient(xgrad, xSum.size() == 0 ? xDiff : Compute::sum(xDiff, xSum));
                }

                if (ygrad != nullptr) {
                    auto yDiff = Compute::matMul(Compute::transpose(*x), g);
                    Gradienter<T, Device>::addGradient(ygrad, ySum.size() == 0 ? yDiff : Compute::sum(yDiff, ySum));
                }
            },
            a, b));
}

/**
 * Calculate power of a by s, will use fast pow optimization if s is integer
 * @tparam T Type
 * @tparam Device Device
 * @tparam S power type
 * @param a a
 * @param s s
 * @return a^s
 */
template <typename T, typename Device, typename S>
auto pow(const Tensor<T, Device> &a, S &&s) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::pow(a.getBase(), s));
    return Tensor<T, Device>(
        result,
        gradientDecider<T, Device>(
            result->size(),
            [x = a.acquireBase(), s, one = ComputationConstant<std::decay_t<S>>::One](auto &&g, auto &&xgrad) mutable {
                if (s != ComputationConstant<std::decay_t<S>>::Zero) {
                    TRY_ADD_GRAD(xgrad,
                                 (Computation<T, Device>::mul(
                                     g, Computation<T, Device>::mul(T(s), Computation<T, Device>::pow(*x, s - one)))));
                }
            },
            a));
}

/**
 * Apply conv2d on two tensor
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param b b
 * @return a conv b
 */
template <typename T, typename Device>
auto conv2d(const Tensor<T, Device> &a, const Tensor<T, Device> &b) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::conv2d(a.getBase(), b.getBase()));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [x = a.acquireBase(), y = b.acquireBase(), kerSize = b.size()](auto &&g, auto &&xgrad,
                                                                                   auto &&ygrad) mutable {
                        TRY_ADD_GRAD(
                            ygrad, (Computation<T, Device>::permute(
                                       Computation<T, Device>::conv2d(Computation<T, Device>::permute(*x, {1, 0, 2, 3}),
                                                                      Computation<T, Device>::permute(g, {1, 0, 2, 3})),
                                       {1, 0, 2, 3})));

                        if (xgrad != nullptr) {
                            auto z = Computation<T, Device>::permute(
                                Computation<T, Device>::flip(Computation<T, Device>::flip(*y, y->dimensionSize() - 1),
                                                             y->dimensionSize() - 2),
                                {1, 0, 2, 3});

                            Gradienter<T, Device>::addGradient(
                                xgrad, Computation<T, Device>::conv2d(
                                           Computation<T, Device>::pad(g,
                                                                       std::vector<std::pair<SizeType, SizeType>>{
                                                                           {0, 0},
                                                                           {0, 0},
                                                                           {kerSize[2] - 1, kerSize[2] - 1},
                                                                           {kerSize[3] - 1, kerSize[3] - 1}}),
                                           z));
                        }
                    },
                    a, b));
}

/**
 * Pad a tensor by specific dim. Example: [[2],[2]] pad on {(0,1), (1, 0)} will be [[0, 2],[0, 2], [0, 0]]
 * @tparam T Type
 * @tparam Device Device
 * @param a tensor
 * @param padder {(left pad, right pad)} * a.dimensionSize()
 * @return padded
 */
template <typename T, typename Device>
auto pad(const Tensor<T, Device> &a, const std::vector<std::pair<SizeType, SizeType>> &padder) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::pad(a.getBase(), padder));

    base::Shape shp = result->size();
    std::vector<std::pair<SizeType, SizeType>> cutRange(shp.size());
    auto n = shp.size();
    for (SizeType i = 0; i < n; i++) {
        cutRange[i] = {padder[i].first, shp[i] - padder[i].second};
    }

    return Tensor<T, Device>(result,
                             gradientDecider<T, Device>(
                                 result->size(),
                                 [x = a.acquireBase(), cutRange = std::move(cutRange)](auto &&g, auto &&xgrad) mutable {
                                     TRY_ADD_GRAD(xgrad, (Computation<T, Device>::cut(g, cutRange)));
                                 },
                                 a));
}

/**
 * Apply functions on all values in a, NOT GRADIENTABLE, CHANGE VALUE OF A
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param f f
 * @return a := f(a)
 */
template <typename T, typename Device>
Tensor<T, Device> &applyOn(Tensor<T, Device> &a, auto &&f) {
    Computation<T, Device>::forEach(a.getBase(), std::forward<decltype(f)>(f));
    return a;
}

/**
 * Apply functions on all values in a, return f(a) NOT GRADIENTABLE
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param f f
 * @return f(a)
 */
template <typename T, typename Device>
auto apply(const Tensor<T, Device> &a, auto &&f) {
    auto result = std::make_shared<BaseTensor<T, Device>>(
        Computation<T, Device>::apply(a.getBase(), std::forward<decltype(f)>(f)));
    return Tensor<T, Device>(result);
}

/**
 * Apply functions on all values in a, providing back prop function back
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param f f
 * @param back b
 * @return a = f(a)
 */
template <typename T, typename Device>
auto apply(const Tensor<T, Device> &a, auto &&f, auto &&back) {
    auto result = std::make_shared<BaseTensor<T, Device>>(
        Computation<T, Device>::apply(a.getBase(), std::forward<decltype(f)>(f)));
    return Tensor<T, Device>(
        result, gradientDecider<T, Device>(
                    result->size(),
                    [x = a.acquireBase(), diff = std::move(back)](auto &&g, auto &&xgrad) mutable {
                        Gradienter<T, Device>::addGradient(
                            xgrad, Computation<T, Device>::mul(g, Computation<T, Device>::apply(*x, diff)));
                    },
                    a));
}

/**
 * Reshape a to other dimension s
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param s s
 * @return reshaped
 */
template <typename T, typename Device>
auto reshape(const Tensor<T, Device> &a, const tensorlib::base::Shape &s) {
    if constexpr (SHAPE_CHECK) {
        SizeType N = 1, M = 1;
        for (auto x : a.size()) {
            N *= x;
        }
        for (auto y : s) {
            M *= y;
        }
        if (N != M) {
            throw std::out_of_range(tensorlib::util::functional::messageOf(
                "Shapes: ", base::shapeToString(a.size()), " and ", base::shapeToString(s), " aren't equivalent size"));
        }
    }
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::reshape(a.getBase(), s));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [x = a.acquireBase(), aShape = a.size()](auto &&g, auto &&xgrad) mutable {
                                             TRY_ADD_GRAD(xgrad, (Computation<T, Device>::reshape(g, aShape)));
                                         },
                                         a));
}

/**
 * Permute a's dimension
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param idx permutation of a's index
 * @return permuted
 */
template <typename T, typename Device>
auto permute(const Tensor<T, Device> &a, const Index &idx) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::permute(a.getBase(), idx));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [idxBack =
                                              [&idx]() {
                                                  Index back(idx.size());
                                                  for (SizeType i = 0, n = idx.size(); i < n; i++) {
                                                      back[idx[i]] = i;
                                                  }
                                                  return back;
                                              }()](auto &&g, auto &&xgrad) {
                                             TRY_ADD_GRAD(xgrad, (Computation<T, Device>::permute(g, idxBack)));
                                         },
                                         a));
}

/**
 * Expand a by some dimension
 * @tparam T Type
 * @tparam Device Device
 * @param a a
 * @param idx idx of expand, {(new expanded dim idx, expand size)}
 * @return expanded tensor
 */
template <typename T, typename Device>
auto expand(const Tensor<T, Device> &a, const auto &idx) {
    auto result = std::make_shared<BaseTensor<T, Device>>(Computation<T, Device>::expand(a.getBase(), idx));
    return Tensor<T, Device>(result, gradientDecider<T, Device>(
                                         result->size(),
                                         [smIdx =
                                              [&idx]() {
                                                  Index r(idx.size());
                                                  for (SizeType i = 0, n = idx.size(); i < n; i++) {
                                                      r[i] = idx[i].first;
                                                  }
                                                  return r;
                                              }()](auto &&g, auto &&xgrad) mutable {
                                             Gradienter<T, Device>::addGradient(xgrad,
                                                                                Computation<T, Device>::sum(g, smIdx));
                                         },
                                         a));
}
}; // namespace functional

#undef TRY_ADD_GRAD
}; // namespace tensor
}; // namespace tensorlib

using namespace tensorlib::tensor::operations;

#endif