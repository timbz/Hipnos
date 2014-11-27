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

#ifndef CUDAMATRIX_H
#define CUDAMATRIX_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common/math/matrix.h"

//col major matrix for cublas
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class CuBlasMatrixSignlePrecision : public Matrix
{
public:
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
    CuBlasMatrixSignlePrecision(int row, int col);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~CuBlasMatrixSignlePrecision();

    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void* data();
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
    std::complex<double> get(int row, int col);
    /**
     * @brief
     *
     * @param row
     * @param col
     * @param e
     */
    /**
     * @brief
     *
     * @param row
     * @param col
     * @param e
     */
    void set(int row, int col, std::complex<double> e);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    Matrix* clone();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    int getCols();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    int getRows();

    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void setDeviceDataChanged();

private:
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void updateHostData();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    void updateDeviceData();

    cuComplex* hostRaw;  
    bool hostDataChanged;  
    cuComplex* deviceRaw;  
    bool deviceDataChanged;  
    int cols, rows;  
};

#endif // CUDAMATRIX_H
