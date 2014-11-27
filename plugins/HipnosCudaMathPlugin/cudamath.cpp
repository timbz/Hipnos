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

#include "cudamath.h"
#include "cublasmatrix.h"
#include "cublasvector.h"

extern "C"
cudaError_t cuda_exp(cuDoubleComplex* data, int size);

extern "C"
cudaError_t cuda_mult_inplace(cuDoubleComplex* data1, cuDoubleComplex* data2, int size);

extern "C"
cudaError_t cuda_mult(cuDoubleComplex* data1, cuDoubleComplex* data2, cuDoubleComplex* result, int size);

extern "C"
cudaError_t cuda_pow(cuDoubleComplex* data1, int size, int pow);

extern "C"
cudaError_t cuda_for_matrix(cuDoubleComplex* fx, cuDoubleComplex* fy, int rows, int cols, double stepx, double stepy);

CudaMath::CudaMath()
{
    // select device with
//    int numDevices, device;
//    cudaError_t errorId = cudaGetDeviceCount(&numDevices);
//    if (errorId != cudaSuccess) {
//        QString error = "cudaGetDeviceCount returned " + QString::number((int)errorId) + "\n-> " + QString::fromAscii(cudaGetErrorString(errorId)) + "\n";
//        qFatal(error.toAscii());
//    }
//    if (numDevices > 1) {
//          int maxMultiprocessors = 0, maxDevice = 0;
//          for (device = 0; device < numDevices; device++) {
//                  cudaDeviceProp properties;
//                  errorId = cudaGetDeviceProperties(&properties, device);
//                  if (error_id != cudaSuccess) {
//                      QString error = "cudaGetDeviceProperties returned " + QString::number((int)errorId) + "\n-> " + QString::fromAscii(cudaGetErrorString(errorId)) + "\n";
//                      qFatal(error.toAscii());
//                  }
//                  if (maxMultiprocessors < properties.multiProcessorCount) {
//                          maxMultiprocessors = properties.multiProcessorCount;
//                          maxDevice = device;
//                  }
//          }
//          cudaSetDevice(maxDevice);
//    }

    status = cublasCreate(&handle);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("CUBLAS initialisation failed");
    }
    plan = 0;
    currentFftPlanSize = QSize(-1,-1);
}

CudaMath::~CudaMath()
{
    cudaDeviceSynchronize();
    status = cublasDestroy(handle);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Error destring cublas handle in ~CudaMath()");
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

QString CudaMath::getName()
{
    return "Cuda Math Plugin";
}

uint CudaMath::getPerformanceHint()
{
    return 90;
}

bool CudaMath::isPlatformSupported()
{
    int deviceCount;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);
    if (errorId != cudaSuccess) {
        QString error = "cudaGetDeviceCount returned " + QString::number((int)errorId) + "\n-> " + QString::fromAscii(cudaGetErrorString(errorId)) + "\n";
        qFatal(error.toAscii());
    }
    if(deviceCount == 0) // no cuda
    {
        return false;
    }
    bool doublePrecision = false;
    for(int dev = 0; dev < deviceCount; ++dev) // cuda found
    {
        cudaDeviceProp deviceProp;
        errorId = cudaGetDeviceProperties(&deviceProp, dev);
        if (errorId != cudaSuccess) {
            QString error = "cudaGetDeviceProperties returned " + QString::number((int)errorId) + "\n-> " + QString::fromAscii(cudaGetErrorString(errorId)) + "\n";
            qFatal(error.toAscii());
        }
        if(deviceProp.major > 1 || deviceProp.minor > 2) // min compute capability 1.3
            doublePrecision = true;
    }
    if(doublePrecision)
    {
        return true;
    }
    else
    {
        return false;
    }
}

MathPlugin::DataType CudaMath::getDataType()
{
    return DT_DOUBLE_PRECISION;
}

Matrix* CudaMath::createMatrix(int row, int col)
{
   return new CuBlasMatrix(row, col);
}

Vector* CudaMath::createVector(int s)
{
    return new CuBlasVector(s);
}

void CudaMath::forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
{
    cuDoubleComplex* FX = static_cast<cuDoubleComplex*>(fx->data());
    cuDoubleComplex* FY = static_cast<cuDoubleComplex*>(fy->data());
    if(cuda_for_matrix(FX, FY, fx->getRows(), fx->getCols(), stepx, stepy) != cudaSuccess)
    {
        qFatal("cuda_for_matrix failed");
    }
    static_cast<CuBlasMatrix*>(fx)->setDeviceDataChanged();
    static_cast<CuBlasMatrix*>(fy)->setDeviceDataChanged();
}

void CudaMath::mult(Matrix* a, Matrix* b, Matrix* c)
{
    cuDoubleComplex alpha = make_cuDoubleComplex(1,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    cuDoubleComplex* B = static_cast<cuDoubleComplex*>(b->data());
    cuDoubleComplex* C = static_cast<cuDoubleComplex*>(c->data());

    status = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         a->getRows(), b->getCols(), a->getCols(),
                         &alpha, A, a->getRows(),
                         B, b->getRows(),
                         &beta, C, c->getRows());

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZgemm failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrix*>(c)->setDeviceDataChanged();
}

void CudaMath::mult(Matrix* a, Vector* x, Vector* y)
{
    cuDoubleComplex alpha = make_cuDoubleComplex(1,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    cuDoubleComplex* X = static_cast<cuDoubleComplex*>(x->data());
    cuDoubleComplex* Y = static_cast<cuDoubleComplex*>(y->data());

    status = cublasZgemv(handle, CUBLAS_OP_N,
                         a->getRows(), a->getCols(),
                         &alpha, A, a->getRows(),
                         X, 1, &beta, Y, 1);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZgemv failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasVector*>(y)->setDeviceDataChanged();
}

void CudaMath::mult(Matrix *a, std::complex<double> s)
{
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    cuDoubleComplex alpha = make_cuDoubleComplex(s.real(), s.imag());
    status = cublasZscal(handle, a->getRows() * a->getCols(), &alpha, A, 1);

    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("cublasZscal failed");
    }
    // set data changed flag on c matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::componentWiseMult(Matrix* a, Matrix* b, Matrix* c)
{
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    cuDoubleComplex* B = static_cast<cuDoubleComplex*>(b->data());
    cuDoubleComplex* C = static_cast<cuDoubleComplex*>(c->data());

    if(cuda_mult(A, B, C, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_mult failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrix*>(c)->setDeviceDataChanged();
}

void CudaMath::componentWiseMult(Matrix* a, Matrix* b)
{
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    cuDoubleComplex* B = static_cast<cuDoubleComplex*>(b->data());

    if(cuda_mult_inplace(A, B, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_mult_inplace failed");
    }

    // set data changed flag on c matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::add(Matrix *a, Matrix *b)
{
     cuDoubleComplex alpha = make_cuDoubleComplex(1,0);
     cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
     cuDoubleComplex* B = static_cast<cuDoubleComplex*>(b->data());
     status = cublasZaxpy(handle, a->getRows() * a->getCols(), &alpha, B, 1, A, 1);

     if(status != CUBLAS_STATUS_SUCCESS)
     {
         qFatal("cublasZaxpy failed");
     }

     // set data changed flag on a matrix
     static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::exp(Matrix *a)
{
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    if(cuda_exp(A, a->getCols()*a->getRows()) != cudaSuccess)
    {
        qFatal("cuda_exp failed");
    }
    // set data changed flag on a matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::pow(Matrix *a, int p)
{
    cuDoubleComplex* A = static_cast<cuDoubleComplex*>(a->data());
    if(cuda_pow(A, a->getCols()*a->getRows(), p) != cudaSuccess)
    {
        qFatal("cuda_pow failed");
    }
    // set data changed flag on a matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::fft(Matrix *a, int direction)
{
    if(currentFftPlanSize != QSize(a->getRows(), a->getCols()))
    {
        if(plan)
        {
            cufftDestroy(plan);
        }
        cufftResult result = cufftPlan2d(&plan, a->getCols(), a->getRows(), CUFFT_Z2Z);
        if(result != CUFFT_SUCCESS)
        {
            QString error = "cufftPlan2d failed (code " + QString::number(result) + ")";
            qFatal(error.toStdString().c_str());
        }
        currentFftPlanSize = QSize(a->getRows(), a->getCols());
    }
    cufftDoubleComplex* data = static_cast<cufftDoubleComplex*>(a->data());
    cufftExecZ2Z(plan, data, data, direction);

    // set data changed flag on a matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::fft(Matrix *a)
{
    fft(a, CUFFT_FORWARD);
}

void CudaMath::ifft(Matrix *a)
{
    fft(a, CUFFT_INVERSE);
    cuDoubleComplex N = make_cuDoubleComplex(1.0/(a->getRows()*a->getCols()),0);
    cublasZscal(handle, a->getRows()*a->getCols(), &N, static_cast<cuDoubleComplex*>(a->data()), 1);
}

void CudaMath::fft(Matrix *in, Matrix *out)
{
}

void CudaMath::ifft(Matrix *in, Matrix *out)
{
}

void CudaMath::fftshift(Matrix *a)
{
    Q_ASSERT(a->getRows() % 2 == 0);
    Q_ASSERT(a->getCols()% 2 == 0);

    int N = a->getRows()/2;
    int M = a->getCols()/2;
    for(int col = 0; col < M; col++)
    {
        cuDoubleComplex* X = &static_cast<cuDoubleComplex*>(a->data())[col * a->getRows()];
        cuDoubleComplex* Y = &static_cast<cuDoubleComplex*>(a->data())[(M + col) * a->getRows() + N];
        cublasZswap(handle, N, X, 1, Y, 1);

        X = &static_cast<cuDoubleComplex*>(a->data())[col * a->getRows() + N];
        Y = &static_cast<cuDoubleComplex*>(a->data())[(M + col) * a->getRows()];
        cublasZswap(handle, N, X, 1, Y, 1);
    }

    // set data changed flag on a matrix
    static_cast<CuBlasMatrix*>(a)->setDeviceDataChanged();
}

void CudaMath::ifftshift(Matrix *a)
{
    fftshift(a);
}

Q_EXPORT_PLUGIN2(hipnoscudamathplugin, CudaMath);
