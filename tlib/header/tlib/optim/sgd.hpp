#ifndef __TLIB__OPTIMIZER_SGD_HPP__
#define __TLIB__OPTIMIZER_SGD_HPP__

#include "tlib/tlib_tensor.hpp"
#include "optimizer.hpp"

namespace tensorlib {
namespace optim {

/**
 * SGD Optimizer
 * @tparam T
 * @tparam Device
 */
template <typename T, typename Device>
class SGD final : public Optimizer<T, Device> {
private:
    T gamma;
    std::vector<tensor::BaseTensor<T, Device>> parameterMomentum;

public:

    /**
     * @param parameters parameters
     * @param lr learning rate
     * @param gamma momentum
     */
    SGD(const typename Optimizer<T, Device>::Parameters &parameters, T lr, T gamma = tensor::ComputationConstant<T>::Zero)
        : Optimizer<T, Device>(parameters, std::move(lr)), gamma(gamma),
          parameterMomentum(
              gamma == tensor::ComputationConstant<T>::Zero
                  ? std::vector<tensor::BaseTensor<T, Device>>()
                  : Optimizer<T, Device>::baseLike(this->parameters, tensor::ComputationConstant<T>::Zero)) {
    }

    void step() override {
        SizeType n = this->parameters.size();
        if (gamma == tensor::ComputationConstant<T>::Zero) {
            for (SizeType i = 0; i < n; i++) {

                Tensor<T, Device> &t = *(this->parameters[i]);
                if (t.isRequireGrad()) {
                    tensor::Computation<T, Device>::minusEqual(
                        t.getBase(), tensor::Computation<T, Device>::mul(t.getGradientBase(), this->lr));
                }
            }
        } else {
            for (SizeType i = 0; i < n; i++) {
                Tensor<T, Device> &t = *(this->parameters[i]);
                if (t.isRequireGrad()) {
                    tensor::BaseTensor<T, Device> &m = parameterMomentum[i];
                    tensor::Computation<T, Device>::mulEqual(m, this->gamma);
                    tensor::Computation<T, Device>::addEqual(m, t.getGradientBase());
                    tensor::Computation<T, Device>::minusEqual(t.getBase(),
                                                               tensor::Computation<T, Device>::mul(m, this->lr));
                }
            }
        }
    }
};

}; // namespace optim
}; // namespace tensorlib

#endif