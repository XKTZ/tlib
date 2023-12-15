#ifndef __TLIB__NN_LINEAR_HPP__
#define __TLIB__NN_LINEAR_HPP__

#include "tlib/nn/module.hpp"

namespace tensorlib {
namespace nn {

/**
 * Linear layer, x * M, M is matrix, x = [..., N, K], M = [K, L]
 * @tparam T Type
 * @tparam Device Device
 * @tparam Biased Bias is performed or not
 */
template <typename T, typename Device, bool Biased = true>
class Linear : public Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>> {

    Tensor<T, Device> mat;
    Tensor<T, Device> bias;

public:

    /**
     * @param dimIn Dimension of Input
     * @param dimOut Dimension of Output
     */
    Linear(SizeType dimIn, SizeType dimOut)
        : Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>>(),
          mat(dimIn, dimOut), bias(dimOut) {
        this->registerParameter(mat);
        if constexpr (Biased) {
            this->registerParameter(bias);
        }
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &input) override {
        if constexpr (Biased) {
            return tensorlib::tensor::functional::matMul(input, mat) + bias;
        } else {
            return tensorlib::tensor::functional::matMul(input, mat);
        }
    }

    std::string_view name() const override {
        if constexpr (Biased) {
            return "Linear with Biased";
        } else {
            return "Linear without Biased";
        }
    }
};

}; // namespace nn
}; // namespace tensorlib

#endif //__TLIB__NN_LINEAR_HPP__
