#ifndef __TLIB__TENSOR_COMPUTATIONAL_HPP__
#define __TLIB__TENSOR_COMPUTATIONAL_HPP__

#include "tlib/tlib_base.hpp"
#include "tlib/tlib_dimensional.hpp"

namespace tensorlib {
namespace tensor {

namespace computational {

/**
 * Constants, that used for incase the data type doesn't have good values can be casted from float (e.g. multiplicative
 * identity)
 * Rules:
 * - Constant<T>::AddictiveIdentity (e.g. 0)
 * - Constant<T>::MultiplicativeIdentity (e.g. 1)
 * - Constant<T>::Zero = Constant<T>::AddictiveIdentity
 * - Constant<T>::One = Constant<T>::MultiplicativeIdentity
 */
template <typename T>
struct ComputationalConstant {
    static constexpr T AddictiveIdentity = 0;
    static constexpr T MultiplicativeIdentity = 1;
    static constexpr T Zero = AddictiveIdentity;
    static constexpr T One = MultiplicativeIdentity;
};

/**
 * Computational is a not defined class used for specialization
 * It should provides following functionality:
 * Computational<Device>::Base<T> => Tensor base created, that extends following:
 * - Indexable<Base<T>>
 * - OffsetDimensional
 * - Gettable<T>
 * For example, if user provides a
 */
template <typename T, typename Device>
struct Computational;

template <typename T, typename Device>
concept IsValidComputationalInBase =
    std::is_base_of_v<tensorlib::base::Indexable<typename Computational<T, Device>::Base>,
                      typename Computational<T, Device>::Base> &&
    std::is_base_of_v<tensorlib::base::OffsetDimensional, typename Computational<T, Device>::Base> &&
    std::is_base_of_v<tensorlib::base::Gettable<T>, typename Computational<T, Device>::Base>;

template <typename T, typename Device>
concept IsValidComputational = IsValidComputationalInBase<T, Device>;
}; // namespace computational

template <typename T, typename Device>
using BaseTensor = typename computational::Computational<T, Device>::Base;

template <typename T, typename Device>
using Computation = tensorlib::tensor::computational::Computational<T, Device>;

template<typename T>
using ComputationConstant = tensorlib::tensor::computational::ComputationalConstant<T>;


}; // namespace tensor
}; // namespace tensorlib

#endif