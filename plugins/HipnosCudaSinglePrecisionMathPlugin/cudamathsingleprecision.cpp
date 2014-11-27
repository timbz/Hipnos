/****************************************************************************
** 
** This file is part of HiPNOS.
** 
** Copyright 2012 Helmholtz-Zentrum Dresden-Rossendorf
** 
** HiPNOS is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** HiPNOS is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with HiPNOS.  If not, see <http://www.gnu.org/licenses/>.
** 
****************************************************************************/


#include "cudamathsingleprecision.h"
extern "C"
cudaError_t cuda_exp(cuComplex* data, int size);

extern "C"
cudaError_t cuda_mult_inplace(cuComplex* data1, cuComplex* data2, int size);

extern "C"
cudaError_t cuda_mult(cuComplex* data1, cuComplex* data2, cuComplex* result, int size);

extern "C"
cudaError_t cuda_pow(cuComplex* data1, int size, int pow);

extern "C"
cudaError_t cuda_for_matrix(cuComplex* fx, cuComplex* fy, int rows, int cols, float stepx, float stepy);

CudaMathSinglePrecision::CudaMathSinglePrecision()
{
    status = cublasCreate(&handle);
    //cudaSetDevice()
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("CUBLAS initialisation failed");
    }
    plan = 0;
    currentFftPlanSize = QSize(-1,-1);
}

CudaMathSinglePrecision::~CudaMathSinglePrecision()
{
    cudaDeviceSynchronize();
    status = cublasDestroy(handle);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Error destring cublas handle in ~CudaMathSinglePrecision()");
    }
    if(plan)
    {
        cufftResult result = cufftDestroy(plan);
        if(result != CUFFT_SUCCESS)
        {
            QString error = "cufftPlan2d failed (code " + QString::number(result) + ")";
            qFatal(error.toStdString().c_str());
        }
    }
}

QString CudaMathSinglePrecision::getName()
{
    return "Cuda Math Plugin (Single Precision)";
}

uint CudaMathSinglePrecision::getPerformanceHint()
{
    return 50;
}

bool CudaMathSinglePrecision::isPlatformSupported()
{
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        QString error = "cudaGetDeviceCount returned " + QString::number((int)error_id) + "\n-> " + QString::fromAscii(cudaGetErrorString(error_id)) + "\n";
        qFatal(error.toAscii());
    }
    if(deviceCount == 0) // no cuda
    {
        return false;
    }
    return true;
}

MathPlugin::DataType CudaMathSinglePrecision::getDataType()
{
    return DT_SINGLE_PRECISION;
}

Matrix* CudaMathSinglePrecision::createMatrix(int row, int col)
{
   return new CuBlasMatrixSignlePrecision(row, col);
}

Vector* CudaMathSinglePrecision::createVector(int s)
{
    return new CuBlasVectorSinglePrecision(s);
}

void CudaMathSinglePrecision::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
{
    cuComplex* FX = static_cast<cuComplex*>(fx->data());
    cuComplex* FY = static_cast<cuComplex*>(fy->data());
    if(cuda_for_matrix(FX, FY, fx->getRows(), fx->getCols(), stepx, stepy) != cudaSuccess)
    {
        qFatal("cuda_for_matrix failed");
    }
    static_cast<CuBlasMatrixSignlePrecision*>(fx)->setDeviceDataChanged();
    static_cast<CuBlasMatrixSignlePrecision*>(fy)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::mult(Matrix* a, Matrix* b, Matrix* c)
{
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex beta = make_cuComplex(0,0);
    cuComplex* A = static_cast<cuComplex*>(a->data());
    cuComplex* B = static_cast<cuComplex*>(b->data());
    cuComplex* C = static_cast<cuComplex*>(c->data());

    status = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         a->getRows(), b->getCols(), a->getCols(),
                         &alpha, A, a->getRows(),
                         B, b->getRows(),
                         &beta, C, c->getRows());

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZgemm failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrixSignlePrecision*>(c)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::mult(Matrix* a, Vector* x, Vector* y)
{
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex beta = make_cuComplex(0,0);
    cuComplex* A = static_cast<cuComplex*>(a->data());
    cuComplex* X = static_cast<cuComplex*>(x->data());
    cuComplex* Y = static_cast<cuComplex*>(y->data());

    status = cublasCgemv(handle, CUBLAS_OP_N,
                         a->getRows(), a->getCols(),
                         &alpha, A, a->getRows(),
                         X, 1, &beta, Y, 1);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZgemv failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasVectorSinglePrecision*>(y)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::mult(Matrix *a, std::complex<double> s)
{
    cuComplex* A = static_cast<cuComplex*>(a->data());
    cuComplex alpha = make_cuComplex(s.real(), s.imag());
    status = cublasCscal(handle, a->getRows() * a->getCols(), &alpha, A, 1);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZscal failed");
    }
    // set data changed flag on c matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
{
    cuComplex* A = static_cast<cuComplex*>(a->data());
    cuComplex* B = static_cast<cuComplex*>(b->data());
    cuComplex* C = static_cast<cuComplex*>(c->data());

    if(cuda_mult(A, B, C, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_mult failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrixSignlePrecision*>(c)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::componentWiseMult(Matrix* a, Matrix* b)
{
    cuComplex* A = static_cast<cuComplex*>(a->data());
    cuComplex* B = static_cast<cuComplex*>(b->data());

    if(cuda_mult_inplace(A, B, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_mult_inplace failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::add(Matrix *a, Matrix *b)
{
     cuComplex alpha = make_cuComplex(1,0);
     cuComplex* A = static_cast<cuComplex*>(a->data());
     cuComplex* B = static_cast<cuComplex*>(b->data());
     status = cublasCaxpy(handle, a->getRows() * a->getCols(), &alpha, B, 1, A, 1);

     if(status != CUBLAS_STATUS_SUCCESS)
     {
         qFatal("cublasCaxpy failed");
     }

     // set data changed flag on a matrix
     static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::exp(Matrix *a)
{
    cuComplex* A = static_cast<cuComplex*>(a->data());
    if(cuda_exp(A, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_exp failed");
    }
    // set data changed flag on a matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::pow(Matrix *a, int p)
{
    cuComplex* A = static_cast<cuComplex*>(a->data());
    if(cuda_pow(A, a->getCols()*a->getRows(), p) != cudaSuccess)
    {
        qFatal("cuda_pow failed");
    }
    // set data changed flag on a matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::fft(Matrix *a, int direction)
{
    if(currentFftPlanSize != QSize(a->getRows(), a->getCols()))
    {
        if(plan)
        {
            cufftDestroy(plan);
        }
        cufftResult result = cufftPlan2d(&plan, a->getCols(), a->getRows(), CUFFT_C2C);
        if(result != CUFFT_SUCCESS)
        {
            QString error = "cufftPlan2d failed (code " + QString::number(result) + ")";
            qFatal(error.toStdString().c_str());
        }
        currentFftPlanSize = QSize(a->getRows(), a->getCols());
    }
    cufftComplex* data = static_cast<cufftComplex*>(a->data());
    cufftExecC2C(plan, data, data, direction);

    // set data changed flag on a matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::fft(Matrix *a)
{
    fft(a, CUFFT_FORWARD);
}

void CudaMathSinglePrecision::ifft(Matrix *a)
{
    fft(a, CUFFT_INVERSE);
    cuComplex N = make_cuComplex(1.0/(a->getRows()*a->getCols()),0);
    cublasCscal(handle, a->getRows()*a->getCols(), &N, static_cast<cuComplex*>(a->data()), 1);
}

void CudaMathSinglePrecision::fft(Matrix *in, Matrix *out)
{
}

void CudaMathSinglePrecision::ifft(Matrix *in, Matrix *out)
{
}

void CudaMathSinglePrecision::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols()% 2 == 0);

    int N = a->getRows()/2;
    int M = a->getCols()/2;
    for(int col = 0; col < M; col++)
    {
        cuComplex* X = &static_cast<cuComplex*>(a->data())[col * a->getRows()];
        cuComplex* Y = &static_cast<cuComplex*>(a->data())[(M + col) * a->getRows() + N];
        cublasCswap(handle, N, X, 1, Y, 1);

        X = &static_cast<cuComplex*>(a->data())[col * a->getRows() + N];
        Y = &static_cast<cuComplex*>(a->data())[(M + col) * a->getRows()];
        cublasCswap(handle, N, X, 1, Y, 1);
    }

    // set data changed flag on a matrix
    static_cast<CuBlasMatrixSignlePrecision*>(a)->setDeviceDataChanged();
}

void CudaMathSinglePrecision::ifftshift(Matrix *a)
{
    fftshift(a);
}

Q_EXPORT_PLUGIN2(hipnoscudamathsingleprecisionplugin, CudaMathSinglePrecision);
