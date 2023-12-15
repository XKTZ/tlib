#ifndef __TLIB__NN_INIT_HPP__
#define __TLIB__NN_INIT_HPP__

#include "tlib/tlib_tensor.hpp"
#include <math.h>
#include <random>

namespace tensorlib {
namespace nn {
namespace init {

using tensorlib::random::manualSeed;

/**
 * Kaiming uniform
 * reference: [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)]
 * @tparam inputDim dimension of input variable is on
 * @tparam T Type
 * @tparam Device Device
 * @param x initializing tensor
 */
template <SizeType inputDim = 0, typename T, typename Device>
void kaimingInitialization(Tensor<T, Device> &x) {
    SizeType r = x.size()[inputDim];
    for (SizeType j = 2; j < x.dimensionSize(); j ++) {
        r *= x.size()[j];
    }
    tensor::Computation<T, Device>::forEach(
        x.getBase(), [dis = std::uniform_real_distribution(-sqrt(T(3) / r), sqrt(T(3) / r))](auto &p) mutable {
            p = dis(tensorlib::random::randomGen);
        });
}

}; // namespace init
}; // namespace nn
}; // namespace tensorlib

#endif