#ifdef _TLIB_OPTIMIZE_CPU_BLAS

#include "tlib/tlib.hpp"
#include "cblas.h"

namespace tensorlib::compute {

using T = tensorlib::compute::Engine<float>::T;

static CBLAS_ORDER matOrderToCblasOrder(ComputeMatrix<T>::MatrixOrder order) {
    switch (order) {
    case ComputeMatrix<float>::COL:
        return CBLAS_ORDER::CblasColMajor;
    case ComputeMatrix<float>::ROW:
        return CBLAS_ORDER ::CblasRowMajor;
    default:
        return CBLAS_ORDER::CblasColMajor;
    };
}

static CBLAS_TRANSPOSE matOperationToCblasOperation(ComputeMatrix<T>::MatrixOperation op) {
    switch (op) {
    case ComputeMatrix<float>::NO_TRANS:
        return CBLAS_TRANSPOSE::CblasNoTrans;
    case ComputeMatrix<float>::TRANS:
        return CBLAS_TRANSPOSE::CblasTrans;
    case ComputeMatrix<float>::CONJ_NO_TRANS:
        return CBLAS_TRANSPOSE::CblasConjNoTrans;
    case ComputeMatrix<float>::CONJ_TRANS:
        return CBLAS_TRANSPOSE::CblasConjTrans;
    default:
        return CBLAS_TRANSPOSE::CblasNoTrans;
    };
}

#ifdef TLIB_CPU_BLAS_ENGINE_S_AXPY
void Engine<float>::axpy(ComputeVector<T> to, T alpha, ComputeVector<T> x) {
    cblas_saxpy(to.n, alpha, x.vec, x.inc, to.vec, to.inc);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_COPY
void Engine<float>::copy(ComputeVector<T> to, ComputeVector<T> x) {
    cblas_scopy(to.n, x.vec, x.inc, to.vec, to.inc);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_SCAL
void Engine<float>::scal(ComputeVector<T> to, T alpha) {
    cblas_sscal(to.n, alpha, to.vec, to.inc);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_DOT
T Engine<float>::dot(ComputeVector<T> x, ComputeVector<T> y) {
    return cblas_sdot(x.n, x.vec, x.inc, y.vec, y.inc);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_GEMV
void Engine<float>::gemv(ComputeMatrix<T>::MatrixOrder order, T beta, ComputeVector<T> to, T alpha,
                         ComputeMatrix<T> mat, ComputeVector<T> x) {
    cblas_sgemv(matOrderToCblasOrder(order), matOperationToCblasOperation(mat.operation), mat.row, mat.col, alpha,
                mat.mat, mat.ld, x.vec, x.inc, beta, to.vec, to.inc);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_GEMM
void Engine<float>::gemm(ComputeMatrix<T>::MatrixOrder order, T beta, ComputeMatrix<T> to, T alpha, ComputeMatrix<T> x,
                         ComputeMatrix<T> y) {
    cblas_sgemm(matOrderToCblasOrder(order), matOperationToCblasOperation(x.operation),
                matOperationToCblasOperation(y.operation), x.row, y.col, x.col, alpha, x.mat, x.ld, y.mat, y.ld, beta,
                to.mat, to.ld);
}
#endif

#ifdef TLIB_CPU_BLAS_ENGINE_S_IMATCOPY
void Engine<float>::imatcopy(ComputeMatrix<T>::MatrixOrder order, ComputeMatrix<T> mat, T alpha,
                             tensorlib::SizeType ld) {
    cblas_simatcopy(matOrderToCblasOrder(order), matOperationToCblasOperation(mat.operation), mat.row, mat.col, alpha,
                    mat.mat, mat.ld, ld);
}
#endif

#endif
};
