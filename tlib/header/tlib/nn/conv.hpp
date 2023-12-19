#ifndef __TLIB__NN_CONV_HPP__
#define __TLIB__NN_CONV_HPP__

#include <utility>

#include "activation.hpp"
#include "module.hpp"

namespace tensorlib {
namespace nn {

/**
 * Conv2d
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class Conv2d : public Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>> {
    Tensor<T, Device> kernel;
    std::pair<SizeType, SizeType> padding;

public:
    /**
     * @param dimIn Dimension of input
     * @param dimOut Dimension of output
     * @param kernel Size of kernel
     * @param padding Padding size
     */
    Conv2d(SizeType dimIn, SizeType dimOut, std::pair<SizeType, SizeType> kernel, std::pair<SizeType, SizeType> padding)
        : Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>>(),
          kernel(dimOut, dimIn, kernel.first, kernel.second), padding(std::move(padding)) {
        this->template registerParameter<1>(this->kernel);
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &input) override {
        return tensorlib::functional::conv2d(
            tensorlib::tensor::functional::pad(
                input, {{0, 0}, {0, 0}, {padding.first, padding.first}, {padding.second, padding.second}}),
            kernel);
    }

    std::string_view name() const override {
        return "Conv 2D";
    }
};

/**
 * Conv Transpose 2D
 * @tparam T Type
 * @tparam Device Device
 */
template <typename T, typename Device>
class ConvTranspose2d
    : public Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>> {
    std::pair<SizeType, SizeType> scale;
    Conv2d<T, Device> conv;

public:
    /**
     * @param scale Scale up in last two dimensions
     * @param dimIn Dimension of input
     * @param dimOut Dimension of output
     * @param kernel Size of kernel
     * @param padding Padding size
     */
    ConvTranspose2d(std::pair<SizeType, SizeType> scale, SizeType dimIn, SizeType dimOut,
                    std::pair<SizeType, SizeType> kernel, std::pair<SizeType, SizeType> padding)
        : Module<T, Device, tensorlib::tensor::Tensor<T, Device>, tensorlib::tensor::Tensor<T, Device>>(),
          scale(std::move(scale)), conv(dimIn, dimOut, kernel, padding) {
        this->registerParameter(this->conv);
    }

    std::string_view name() const override {
        return "Conv Transpose 2D";
    }

    Tensor<T, Device> forward(const Tensor<T, Device> &input) override {
        auto w = nn::functional::relu(conv(input));

        // input = [..., C, H, W] => [..., C, H, S1, W, S2]
        auto x = tensorlib::functional::expand(
            w, std::array<std::pair<SizeType, SizeType>, 2>{
                       std::pair<SizeType, SizeType>(input.dimensionSize() - 1, scale.first),
                       std::pair<SizeType, SizeType>(input.dimensionSize() + 1, scale.second)});
        x.contiguous();
        auto rshp = w.size();
        rshp[rshp.size() - 2] *= scale.first;
        rshp[rshp.size() - 1] *= scale.second;

        x = tensorlib::functional::reshape(x, rshp);
        return x;
    }
};
}; // namespace nn
}; // namespace tensorlib

#endif