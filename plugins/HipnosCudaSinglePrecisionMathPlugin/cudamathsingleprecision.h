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

#ifndef CUDAMATH_H
#define CUDAMATH_H

#include "common/math/mathplugin.h"
#include "cublasmatrixsingleprecision.h"
#include "cublasvectorsingleprecision.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <QObject>
#include <QtGui>

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class CudaMathSinglePrecision : public QObject, public MathPlugin
{
    Q_OBJECT
    Q_INTERFACES(MathPlugin)

public:
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
    CudaMathSinglePrecision();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~CudaMathSinglePrecision();
    /**
     * @brief
     *
     * @param row
     * @param col
     */
    /**
     * @brief
     *
     * @param row
     * @param col
     */
    Matrix* createMatrix(int row, int col);
    /**
     * @brief
     *
     * @param s
     */
    /**
     * @brief
     *
     * @param s
     */
    Vector* createVector(int s);
    /**
     * @brief
     *
     * @param fx
     * @param fy
     * @param stepx
     * @param stepy
     */
    /**
     * @brief
     *
     * @param fx
     * @param fy
     * @param stepx
     * @param stepy
     */
    void forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy);
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param c
     */
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param c
     */
    void mult(Matrix* a, Matrix* b, Matrix* c);
    /**
     * @brief
     *
     * @param a
     * @param x
     * @param y
     */
    /**
     * @brief
     *
     * @param a
     * @param x
     * @param y
     */
    void mult(Matrix* a, Vector* x, Vector* y);
    /**
     * @brief
     *
     * @param a
     * @param s
     */
    /**
     * @brief
     *
     * @param a
     * @param s
     */
    void mult(Matrix* a, std::complex<double> s);
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param c
     */
    /**
     * @brief
     *
     * @param a
     * @param b
     * @param c
     */
    void componentWiseMult(Matrix* a, Matrix* b, Matrix* c);
    /**
     * @brief
     *
     * @param a
     * @param b
     */
    /**
     * @brief
     *
     * @param a
     * @param b
     */
    void componentWiseMult(Matrix *a, Matrix *b);
    /**
     * @brief
     *
     * @param a
     * @param b
     */
    /**
     * @brief
     *
     * @param a
     * @param b
     */
    void add(Matrix *a, Matrix *b);
    /**
     * @brief
     *
     * @param a
     */
    /**
     * @brief
     *
     * @param a
     */
    void exp(Matrix *a);
    /**
     * @brief
     *
     * @param a
     * @param p
     */
    /**
     * @brief
     *
     * @param a
     * @param p
     */
    void pow(Matrix *a, int p);
    /**
     * @brief
     *
     * @param a
     * @param direction
     */
    /**
     * @brief
     *
     * @param a
     * @param direction
     */
    void fft(Matrix *a, int direction);
    /**
     * @brief
     *
     * @param a
     */
    /**
     * @brief
     *
     * @param a
     */
    void fft(Matrix *a);
    /**
     * @brief
     *
     * @param a
     */
    /**
     * @brief
     *
     * @param a
     */
    void ifft(Matrix *a);
    /**
     * @brief
     *
     * @param in
     * @param out
     */
    /**
     * @brief
     *
     * @param in
     * @param out
     */
    void fft(Matrix* in, Matrix* out);
    /**
     * @brief
     *
     * @param in
     * @param out
     */
    /**
     * @brief
     *
     * @param in
     * @param out
     */
    void ifft(Matrix* in, Matrix* out);
    /**
     * @brief
     *
     * @param a
     */
    /**
     * @brief
     *
     * @param a
     */
    void fftshift(Matrix *a);
    /**
     * @brief
     *
     * @param a
     */
    /**
     * @brief
     *
     * @param a
     */
    void ifftshift(Matrix *a);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    QString getName();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    uint getPerformanceHint();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    bool isPlatformSupported();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    DataType getDataType();

private:
    cublasStatus_t status;  
    cublasHandle_t handle;  
    QSize currentFftPlanSize;  
    cufftHandle plan;  
};
#endif // CUDAMATH_H
