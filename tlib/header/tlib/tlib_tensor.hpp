#ifndef __TLIB__TENSOR_ALL_HPP__
#define __TLIB__TENSOR_ALL_HPP__

#include "tlib/tensor/cpu/tensor_cpu.hpp"
#include "tlib/tensor/tensor.hpp"
#include "tlib/tensor/tensor_operations.hpp"

namespace tensorlib {
using tensor::Tensor;
namespace functional {
using namespace tensorlib::tensor::functional;
};
}; // namespace tensorlib

#endif