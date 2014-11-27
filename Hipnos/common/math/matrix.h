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

#ifndef MATRIX_H
#define MATRIX_H

#include <complex>

/**
 * @brief Interface to a matrix type. Each plugin has to implements a specific concretisation
 *
 */
class Matrix
{

public:
    /**
     * @brief Destructor
     *
     */
    virtual ~Matrix() {}

    /**
     * @brief Gets the number of rows
     *
     */
    virtual int getRows() = 0;

    /**
     * @brief Gets the number of columns
     *
     */
    virtual int getCols() = 0;

    /**
     * @brief Returns a void pointer to the raw data
     *
     */
    virtual void* data() = 0;

    /**
     * @brief Gets the value of a specific element of the Matrix
     *
     * @param row Row index of the element
     * @param col Column index of the element
     */
    virtual std::complex<double> get(int row, int col) = 0;

    /**
     * @brief Sets the value of a specific element of the Matrix
     *
     * @param row Row index of the element
     * @param col Column index of the element
     * @param e New value of the element
     */
    virtual void set(int row, int col, std::complex<double> e) = 0;

    /**
     * @brief Returns a clone of this Matrix
     *
     */
    virtual Matrix* clone() = 0;

};

#endif // MATRIX_H
