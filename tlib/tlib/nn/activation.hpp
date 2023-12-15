#ifndef __TLIB__NN_RELU__HPP
#define __TLIB__NN_RELU__HPP

#include "tlib/nn/module.hpp"

namespace tensorlib {
namespace nn {

namespace functional {

/**
 * ReLU, relu(x) = max(x, 0)
 * @tparam T Type
 * @tparam Device Device
 * @param x input tensor
 * @return relu(x)
 */
template <typename T, typename Device>
auto relu(const Tensor<T, Device> &x) {
    return tensorlib::functional::apply(
        x,
        [](auto &&x) {
            return x > tensor::computational::ComputationalConstant<T>::Zero
                       ? x
                       : tensor::computational::ComputationalConstant<T>::Zero;
        },
        [](auto &&x) {
            return x > tensor::computational::ComputationalConstant<T>::Zero
                       ? tensor::computational::ComputationalConstant<T>::One
                       : tensor::computational::ComputationalConstant<T>::Zero;
        });
}

/**
 * Leaky ReLU, leakyrelu(x, alpha) = max(x, x * alpha)
 * @tparam T Type
 * @tparam Device Device
 * @param x x
 * @param alpha alpha
 * @return leakyrelu(x, alpha)
 */
template <typename T, typename Device>
auto leakyRelu(const Tensor<T, Device> &x, const T &alpha) {
    return tensorlib::functional::apply(
        x, [alpha](auto &&x) { return x > tensor::computational::ComputationalConstant<T>::Zero ? x : alpha * x; },
        [alpha](auto &&x) {
            return x > tensor::computational::ComputationalConstant<T>::Zero
                       ? tensor::computational::ComputationalConstant<T>::One
                       : alpha;
        });
}

/**
 * Softmax x along dimension
 * @tparam T Type
 * @tparam Device Device
 * @param x x
 * @param idx softmax dimension
 * @return softmax(x)
 */
template <typename T, typename Device>
auto softmax(const Tensor<T, Device> &x, const Index &idx) {
    Tensor<T, Device> ex = tensorlib::functional::exp(x);

    SizeType n = idx.size();
    std::vector<std::pair<SizeType, SizeType>> expander(n);
    Shape shp = x.size();
    for (SizeType i = 0; i < n; i++) {
        expander[i] = {idx[i], shp[idx[i]]};
    }

    Tensor<T, Device> sm = tensorlib::functional::expand(tensorlib::functional::sum(ex, idx), expander);

    return ex / sm;
}

/**
 * This is the preserved code for softmax with optimize that is not used for now
 */
// template <typename T, typename Device>
// auto softmax(const Tensor<T, Device> &x, auto &&) {
//     Tensor<T, Device> ex = tensorlib::functional::exp(
//         x - tensorlib::functional::expand(tensorlib::functional::maxBack(x),
//                                           std::array<std::pair<SizeType, SizeType>, 1>{
//                                               std::pair<SizeType, SizeType>{x.dimensionSize() - 1,
//                                               x.size().back()}}));
//
//     Tensor<T, Device> sm =
//         tensorlib::functional::expand(tensorlib::functional::sum(ex, {x.dimensionSize() - 1}),
//                                       std::array<std::pair<SizeType, SizeType>, 1>{
//                                           std::pair<SizeType, SizeType>{x.dimensionSize() - 1, x.size().back()}});
//
//     return ex / sm;
// }

}; // namespace functional

/**
 * ReLU, relu(x) = max(x, 0)
 * @tparam T T
 * @tparam Device Device
 */
template <typename T, typename Device>
class ReLU : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>> {

public:
    ReLU() : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>>() {
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &x) override {
        return functional::relu(x);
    }

    std::string_view name() const override {
        return "relu";
    }
};

/**
 * Leaky ReLU, leakyrelu(x, alpha) = max(x, x * alpha)
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class LeakyReLU : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>> {
    T alpha;

public:
    /**
     * @param alpha alpha
     */
    explicit LeakyReLU(T alpha = T(0.01)) : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>>(), alpha(alpha) {
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &x) override {
        return functional::leakyRelu(x, alpha);
    }

    std::string_view name() const override {
        return "Leaky Relu";
    }
};

template <typename T, typename Device>
class BatchNorm2d : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>> {
    SizeType c;
    Tensor<T, Device> gamma;
    Tensor<T, Device> beta;
    T eps;

public:
    explicit BatchNorm2d(SizeType c, T eps = 1e-5)
        : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>>(), c(c),
          gamma(c, tensor::ComputationConstant<T>::One), beta(c, tensor::ComputationConstant<T>::Zero), eps(eps) {
        this->registerAll(gamma, beta);
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &x) override {
        using pr = std::pair<SizeType, SizeType>;
        using vec = std::vector<pr>;
        SizeType shp = x.size()[0];
        for (SizeType i = 2; i < x.dimensionSize(); i++) {
            shp *= x[i];
        }
        auto mean = tensorlib::functional::expand(tensorlib::functional::sum(x, {0, 2, 3}),
                                                  vec{pr{2, x.size()[2]}, pr{3, x.size()[3]}})
                        .contiguous() /
                    T(shp);
        auto sigma = tensorlib::functional::expand(
            tensorlib::functional::pow(
                tensorlib::functional::sum(tensorlib::functional::pow(x - mean, 2), {0, 2, 3}) / T(shp) + eps, 0.5),
            vec{pr{2, x.size()[2]}, pr{3, x.size()[3]}});

        return (x - mean) / sigma * tensorlib::functional::expand(gamma, vec{pr{1, x.size()[2]}, pr{2, x.size()[3]}}) +
               tensorlib::functional::expand(beta, vec{pr{1, x.size()[2]}, pr{2, x.size()[3]}});
    }
};
}; // namespace nn
}; // namespace tensorlib

#endif