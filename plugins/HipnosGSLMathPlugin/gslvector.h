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

#ifndef CBLASVECTOR_H
#define CBLASVECTOR_H

#include "common/math/vector.h"

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class GslVector : public Vector
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
    GslVector(int s);
    /**
     * @brief
     *
     */
    /**
     * @brief
     *
     */
    ~GslVector();

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

private:
    std::complex<double>* raw;  
    int size;  
};

#endif // CBLASVECTOR_H
