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

#ifndef CUDAVECTOR_H
#define CUDAVECTOR_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "common/math/vector.h"

// vector wrapper for cublas
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class CuBlasVector : public Vector
{
public:
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
    CuBlasVector(int s);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~CuBlasVector();

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
     * @param i
     */
    /**
     * @brief
     *
     * @param i
     */
    std::complex<double> get(int i);
    /**
     * @brief
     *
     * @param i
     * @param s
     */
    /**
     * @brief
     *
     * @param i
     * @param s
     */
    void set(int i, std::complex<double> s);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    Vector* clone();
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    int getSize();

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

    cuDoubleComplex* hostRaw;  
    bool hostDataChanged;  
    cuDoubleComplex* deviceRaw;  
    bool deviceDataChanged;  
    int size;  
};

#endif // CUDAVECTOR_H
