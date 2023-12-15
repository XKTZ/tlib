#ifndef __TLIB__CONFIG_HPP__
#define __TLIB__CONFIG_HPP__

#include <cstdlib>

/**
 * This file operates the macro-provided configurations, such as optimization
*/
namespace tensorlib {

// =============== default configs ===============

// size type
#ifndef _TLIB_CONFIG_SIZE_TYPE
#define _TLIB_CONFIG_SIZE_TYPE size_t
#endif

// out of bound check
#ifndef _TLIB_CONFIG_BOUND_CHECK
#define _TLIB_CONFIG_BOUND_CHECK (true)
#endif

#ifndef _TLIB_CONFIG_SHAPE_CHECK
#define _TLIB_CONFIG_SHAPE_CHECK (true)
#endif

// OPTIMIZE: treat new as noexcept
//   that is, there will be no try-catch (and function would be noexcept)
//   if new/operator new is the only possible exception cause
#ifndef _TLIB_OPTIMIZE_NEW_NOEXCEPT
#define _TLIB_OPTIMIZE_NEW_NOEXCEPT (false)
#endif

// OPTIMIZE: Will use BLAS for the floating point calculation in cpu::TensorBase<float>
//   default yes
#ifndef _TLIB_OPTIMIZE_CPU_BLAS
#define _TLIB_OPTIMIZE_CPU_BLAS 1
#endif

// OPTIMIZE: If BLAS is opened, then computation will check if the size of cpu::TensorBase<float> is greater than
//   _TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS, if so, it will force contiguous the TensorBase
#ifndef _TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS
#define _TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS 1024
#endif

// =============== create config variables ===============

using SizeType = size_t;

constexpr bool OUT_OF_RANGE_CHECK = _TLIB_CONFIG_BOUND_CHECK;

constexpr bool SHAPE_CHECK = _TLIB_CONFIG_SHAPE_CHECK;

constexpr bool NEW_NOEXCEPT = _TLIB_OPTIMIZE_NEW_NOEXCEPT;

constexpr SizeType BLAS_CONTIGUOUS = _TLIB_OPTIMIZE_CPU_BLAS_CONTIGUOUS;

// release the macros
#undef _TLIB_CONFIG_SIZE_TYPE
#undef _TLIB_CONFIG_BOUND_CHECK
#undef _TLIB_OPTIMIZE_NEW_NOEXCEPT
};

#endif