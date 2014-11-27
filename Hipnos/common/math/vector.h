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

#ifndef VECTOR_H
#define VECTOR_H

#include <complex>

/**
 * @brief Interface to a vector type. Each plugin has to implements a specific concretisation
 *
 *
 */
class Vector
{
public:
    /**
     * @brief Destructor
     *
     */
    virtual ~Vector() {}

    /**
     * @brief returns a void pointer to the raw data.
     *
     */
    virtual void* data() = 0;

    /**
     * @brief Gets the value of a specific element of the Vector.
     *
     * @param i Index of the element
     */
    virtual std::complex<double> get(int i) = 0;

    /**
     * @brief Sets the value of a specific element of the Vector
     *
     * @param i Index of the element
     * @param s New value of the element
     */
    virtual void set(int i, std::complex<double> s) = 0;

    /**
     * @brief Returns a clone of this Vector
     *
     */
    virtual Vector* clone() = 0;

    /**
     * @brief Returns the size of the Vector
     *
     */
    virtual int getSize() = 0;
};

#endif // VECTOR_H
