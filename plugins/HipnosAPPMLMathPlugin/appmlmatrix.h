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

#ifndef APPMLMATRIX_H
#define APPMLMATRIX_H

#include <QString>
#include <clAmdBlas.h>

#include "common/math/matrix.h"

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class AppmlMatrix : public Matrix
{
public:
/**
 * @brief
 *
 * @param c
 * @param q
 * @param row
 * @param col
 */
/**
 * @brief
 *
 * @param c
 * @param q
 * @param row
 * @param col
 */
    AppmlMatrix(cl_context c, cl_command_queue q, int row, int col);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~AppmlMatrix();

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
     * @param s
     */
    /**
     * @brief
     *
     * @param row
     * @param col
     * @param s
     */
    void set(int row, int col, std::complex<double> s);
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
    /**
     * @brief
     *
     * @param functionName
     */
    /**
     * @brief
     *
     * @param functionName
     */
    void checkForError(QString functionName);

    std::complex<double>* hostRaw;  
    bool hostDataChanged;  
    cl_mem deviceRaw;  
    bool deviceDataChanged;  
    int rows, cols;  
    int err;  
    cl_context ctx;  
    cl_command_queue queue;  
    cl_event event;  
};


#endif // APPMLMATRIX_H
