#ifndef __TLIB__NN_LOSS_HPP__
#define __TLIB__NN_LOSS_HPP__

#include "tlib/nn/module.hpp"

namespace tensorlib {
namespace nn {

/**
 * L1 Loss, mean(|pred - targ|)
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class L1Loss : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>> {
public:
    L1Loss() : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>>() {
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &pred, const Tensor<T, Device> &target) override {
        return tensor::functional::sum(tensor::functional::abs(pred, target)) / pred.size()[0];
    }

    std::string_view name() const override {
        return "L1 Loss";
    }
};

/**
 * L2 Loss, mean((pred - targ)^2)
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class L2Loss : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>> {
public:
    L2Loss() : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>>() {
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &pred, const Tensor<T, Device> &target) override {
        return tensor::functional::sum(tensor::functional::pow(pred - target, 2)) / pred.size()[0];
    }

    std::string_view name() const override {
        return"L2 Loss";
    }
};

template<typename T, typename Device>
using MSELoss = L2Loss<T,Device>;

/**
 * Cross Entropy Loss, mean(target *log(pred + epsilon))
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class CrossEntropyLoss : public Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>> {
    T epsilon;

public:

    /**
     * @param epsilon epsilon used for avoid overflow
     */
    explicit CrossEntropyLoss(T epsilon = 1e-9)
        : Module<T, Device, Tensor<T, Device>, Tensor<T, Device>, Tensor<T, Device>>(), epsilon(epsilon) {
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &pred, const Tensor<T, Device> &target) override {
        return -tensor::functional::sum(target * tensor::functional::log(pred + epsilon)) / T(pred.size()[0]);
    }

    std::string_view name() const override {
        return "Cross Entropy Loss";
    }
};
}; // namespace nn
}; // namespace tensorlib

#endif
