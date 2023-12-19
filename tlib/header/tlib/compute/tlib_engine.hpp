#ifndef __TLIB__COMPUTE_FLOAT_ENGINE_HPP__
#define __TLIB__COMPUTE_FLOAT_ENGINE_HPP__

#include "tlib/tlib_base.hpp"

#define TLIB_ENGINE_S_AXPY
#define TLIB_ENGINE_S_SCAL
#define TLIB_ENGINE_S_COPY
#define TLIB_ENGINE_S_DOT
#define TLIB_ENGINE_S_GEMV
#define TLIB_ENGINE_S_GEMM
#define TLIB_ENGINE_S_IMATCOPY
#include "tlib/compute/tlib_blas.hpp"

namespace tensorlib {

namespace compute {

template <typename T>
struct ComputeVector {
    SizeType n;
    T *vec;
    SizeType inc;
};

template <typename T>
struct ComputeMatrix {
    enum MatrixOrder { ROW, COL };
    enum MatrixOperation { NO_TRANS, TRANS, CONJ_TRANS, CONJ_NO_TRANS };

    MatrixOperation operation;
    SizeType row;
    SizeType col;
    T *mat;
    SizeType ld;
};

template <typename T>
struct Engine;

template <>
struct Engine<float> {
    using T = float;

#ifndef TLIB_ENGINE_S_AXP
    static void axpy(ComputeVector<T> to, T alpha, ComputeVector<T> x);
#endif

#ifndef TLIB_ENGINE_S_SCAL
    static void scal(ComputeVector<T> to, T alpha);
#endif

#ifndef TLIB_ENGINE_S_COPY
    static void copy(ComputeVector<T> to, ComputeVector<T> x);
#endif

#ifndef TLIB_ENGINE_S_DOT
    static T dot(ComputeVector<T> x, ComputeVector<T> y);
#endif

#ifndef TLIB_ENGINE_S_GEMV
    static void gemv(ComputeMatrix<T>::MatrixOrder order, T beta, ComputeVector<T> to, T alpha, ComputeMatrix<T> mat,
                     ComputeVector<T> x);
#endif

#ifndef TLIB_ENGINE_S_GEMM
    static void gemm(ComputeMatrix<T>::MatrixOrder order, T beta, ComputeMatrix<T> to, T alpha, ComputeMatrix<T> x,
                     ComputeMatrix<T> y);
#endif

#ifndef TLIB_ENGINE_S_IMATCOPY
    static void imatcopy(ComputeMatrix<T>::MatrixOrder order, ComputeMatrix<T> mat, T alpha, SizeType ld);
#endif
};

}; // namespace compute

}; // namespace tensorlib

#endif
