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

#ifndef MATHPLUGIN_H
#define MATHPLUGIN_H

#include <QObject>
#include <QPair>
#include "matrix.h"
#include "vector.h"

/**
 * @brief Interface for platform specific plugin implementations
 *
 */
class MathPlugin
{

public:
    /**
     * @brief Plugin precision (fload or double)
     *
     */
    enum DataType
    {
        DT_SINGLE_PRECISION,
        DT_DOUBLE_PRECISION
    };

    /**
     * @brief Destructor
     *
     */
    virtual ~MathPlugin() {}

    /**
     * @brief Creates a plugin specific Matrix
     *
     * @param row Number of rows
     * @param col Number of columns
     */
    virtual Matrix* createMatrix(int row, int col) = 0;

    /**
     * @brief Creates a plugin specific Vector
     *
     * @param s Size of the Vector
     */
    virtual Vector* createVector(int s) = 0;

    /**
     * @brief Fills each row of fx with values from -fx->getCols()/2*stepx to fx->getCols()/2*stepx
     *          and each column of fy with values from -fy->getRows()/2*stepy to fy->getRwos()/2*stepy.
     *
     * This function can be used to avoid nested loops that iterate over all Matrix elements.
     * Its inspired by the MATLAB meshgrid function
     *
     * @param fx Pointer to the Matrix that will contain the x values of the grid
     * @param fy Pointer to the Matrix that will contain the y values of the grid
     * @param stepx Horizontal sampling step
     * @param stepy Vertical sampling step
     */
    virtual void forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy) = 0;

    // inplace

    /**
     * @brief Out of place Matrix Matrix component wise multiplication. The performed operation is c = a *. b
     *
     * @param a Pointer to the first Matrix
     * @param b Pointer to the second Matrix
     * @param c Pointer to the result Matrix
     */
    virtual void componentWiseMult(Matrix* a, Matrix* b) = 0;

    /**
     * @brief Inplace operation that computes elementwise scalar multiplication.
     *
     * @param a The matrix
     * @param s The scalar value
     */
    virtual void mult(Matrix* a, std::complex<double> s) = 0;

    /**
     * @brief Inplace operation that computes elementwise addition. The performed operation is a += b
     *
     * @param a A pointer to the first Matrix witch will be overwritten
     * @param b A pointer to the second Matrix
     */
    virtual void add(Matrix* a, Matrix* b) = 0;

    /**
     * @brief Inplace operation that applies exponentiation to each element of the Matrix.
     *
     * @param a The Matrix containing the base
     * @param p The exponent
     */
    virtual void exp(Matrix* a) = 0;

    /**
     * @brief Inplace operation that applies exponentiation to each element of the Matrix.
     *
     * @param a The Matrix containing the base
     * @param p The exponent
     */
    virtual void pow(Matrix* a, int p) = 0;

    /**
     * @brief Inplace 2D fourier transformation
     *
     * @param m
     */
    virtual void fft(Matrix* a) = 0;

    /**
     * @brief Inplace inverse 2D fourier transformation
     *
     * @param m
     */
    virtual void ifft(Matrix* a) = 0;

    /**
     * @brief Rearranges the outputs of fft by moving the zero-frequency component to the center of the Matrix
     *
     * @param m
     */
    virtual void fftshift(Matrix* a) = 0;

    /**
     * @brief Reverts the fftshift operation
     *
     * @param m
     */
    virtual void ifftshift(Matrix* a) = 0;

    // out of place
    /**
     * @brief Out of place Matrix Matrix component wise multiplication. The performed operation is c = a *. b
     *
     * @param a Pointer to the first Matrix
     * @param b Pointer to the second Matrix
     * @param c Pointer to the result Matrix
     */
    virtual void componentWiseMult(Matrix* a, Matrix* b, Matrix* c) = 0;

    /**
     * @brief Out of place Matrix Matrix multiplication. The performed operation is c = a * b
     *
     * @param a A pointer to the first Matrix
     * @param b A pointer to the second Matrix
     * @param c A Pointer to the result Matrix
     */
    virtual void mult(Matrix* a, Matrix* b, Matrix* c) = 0;

    /**
     * @brief Out of place Matrix Vector multiplication. The performed operation is y = a * x
     *
     * @param a A pointer to the Matrix
     * @param x A pointer to the input Vector
     * @param y A pointer to the result Vector
     */
    virtual void mult(Matrix* a, Vector* x, Vector* y) = 0;

    /**
     * @brief Out of place 2D fourier transformation.
     *
     * @param in
     * @param out
     */
    virtual void fft(Matrix* in, Matrix* out) = 0;

    /**
     * @brief Out of place inverse 2D fourier transformation.
     *
     * @param in
     * @param out
     */
    virtual void ifft(Matrix* in, Matrix* out) = 0;

    /**
     * @brief Get the unique plugin name
     *
     */
    virtual QString getName() = 0;

    /**
     * @brief Gets a number reflecting the performance of the plugins.
     *
     */
    virtual uint getPerformanceHint() = 0;

    /**
     * @brief Returns true if the current platform is supported by the plugin
     *
     */
    virtual bool isPlatformSupported() = 0;

    /**
     * @brief Returns the precision used by the plugins (float or double)
     *
     */
    virtual DataType getDataType() = 0;
};

Q_DECLARE_INTERFACE(MathPlugin, "org.hipnos.plugin.MathPlugin/1.0");

#endif // MATHPLUGIN_H
