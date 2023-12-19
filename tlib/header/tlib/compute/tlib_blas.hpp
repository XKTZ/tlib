#ifndef __TLIB__COMPUTE_BLAS_HPP__
#define __TLIB__COMPUTE_BLAS_HPP__

#if _TLIB_OPTIMIZE_CPU_BLAS

#include "tlib_engine.hpp"

#ifdef TLIB_ENGINE_S_AXPY
#undef TLIB_ENGINE_S_AXPY
#define TLIB_CPU_BLAS_ENGINE_S_AXPY
#endif

#ifdef TLIB_ENGINE_S_SCAL
#undef TLIB_ENGINE_S_SCAL
#define TLIB_CPU_BLAS_ENGINE_S_SCAL
#endif

#ifdef TLIB_ENGINE_S_COPY
#undef TLIB_ENGINE_S_COPY
#define TLIB_CPU_BLAS_ENGINE_S_COPY
#endif

#ifdef TLIB_ENGINE_S_DOT
#undef TLIB_ENGINE_S_DOT
#define TLIB_CPU_BLAS_ENGINE_S_DOT
#endif

#ifdef TLIB_ENGINE_S_GEMV
#undef TLIB_ENGINE_S_GEMV
#define TLIB_CPU_BLAS_ENGINE_S_GEMV
#endif

#ifdef TLIB_ENGINE_S_GEMM
#undef TLIB_ENGINE_S_GEMM
#define TLIB_CPU_BLAS_ENGINE_S_GEMM
#endif

#ifdef TLIB_ENGINE_S_IMATCOPY
#undef TLIB_ENGINE_S_IMATCOPY
#define TLIB_CPU_BLAS_ENGINE_S_IMATCOPY
#endif

#endif

#endif // __TLIB__COMPUTE_BLAS_HPP__