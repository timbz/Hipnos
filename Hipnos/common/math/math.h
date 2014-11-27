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

#ifndef MATH_H
#define MATH_H

#include "mathplugin.h"
#include <QDebug>

/**
 * @brief This class is the public interface to the math plugin module.
 *
 *      The static methods act a proxy between the application and the loaded plugin.
 *      Internally it is a singelton class with a referece to a MathPlugin implementation.
 */
class Math
{

private:
    /**
     * @brief Private constructor. This class isn t supposed to be instantiated
     *
     */
    Math(){}
    /**
     * @brief Private destructor. This class isn t supposed to be instantiated
     *
     */
    ~Math(){}

    static MathPlugin* instance;  
    static QList<MathPlugin*> mathPlugins;  
    static bool initialised;  

public:
    // this is implemented in the cpp file
    // so we can moc it in the test projects
    /**
     * @brief Initialises the internal MathPlugin singelton.
     *
     */
    static void init();

    /**
     * @brief Creates a plugin specific Matrix implementation
     *
     * @param row Number of rows
     * @param col Number of columns
     * @return Matrix *
     */
    static Matrix* createMatrix(int row, int col)
    {
        init();
        return instance->createMatrix(row, col);
    }

    /**
     * @brief Creates a plugin specific Vector implementation
     *
     * @param s Size of the vector
     * @return Vector *
     */
    static Vector* createVector(int s)
    {
        init();
        return instance->createVector(s);
    }

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
    static void forMatrices(Matrix* fx, Matrix* fy, double stepx, double stepy)
    {
        init();
        if(fx->getCols() != fy->getCols()) return;
        if(fx->getRows() != fy->getRows()) return;
        return instance->forMatrices(fx, fy, stepx, stepy);
    }

    /**
     * @brief Inplace operation that applies the exponential function to each element of the Matrix.
     *
     * @param a The Matrix
     */
    static void exp(Matrix* a)
    {
        init();
        instance->exp(a);
    }

    /**
     * @brief Inplace operation that applies exponentiation to each element of the Matrix.
     *
     * @param a The Matrix containing the base
     * @param p The exponent
     */
    static void pow(Matrix* a, int p)
    {
        init();
        instance->pow(a, p);
    }

    /**
     * @brief Inplace operation that computes elementwise addition. The performed operation is a += b
     *
     * @param a A pointer to the first Matrix witch will be overwritten
     * @param b A pointer to the second Matrix
     */
    static void add(Matrix* a, Matrix* b)
    {
        init();
        if(a->getCols() != b->getCols()) return;
        if(a->getRows() != b->getRows()) return;
        instance->add(a, b);
    }

    /**
     * @brief Out of place Matrix Matrix multiplication. The performed operation is c = a * b
     *
     * @param a A pointer to the first Matrix
     * @param b A pointer to the second Matrix
     * @param c A Pointer to the result Matrix
     */
    static void mult(Matrix* a, Matrix* b, Matrix* c)
    {
        init();
        if(a->getCols() != b->getRows()) return;
        if(a->getRows() != c->getRows()) return;
        if(b->getCols() != c->getCols()) return;
        instance->mult(a,b,c);
    }

    /**
     * @brief Out of place Matrix Vector multiplication. The performed operation is y = a * x
     *
     * @param a A pointer to the Matrix
     * @param x A pointer to the input Vector
     * @param y A pointer to the result Vector
     */
    static void mult(Matrix* a, Vector* x, Vector* y)
    {
        init();
        if(a->getCols() != x->getSize()) return;
        if(a->getRows() != y->getSize()) return;
        instance->mult(a,x,y);
    }

    /**
     * @brief Inplace operation that computes elementwise scalar multiplication.
     *
     * @param a The matrix
     * @param s The scalar value
     */
    static void mult(Matrix* a, std::complex<double> s)
    {
        init();
        instance->mult(a,s);
    }

    /**
     * @brief Out of place Matrix Matrix component wise multiplication. The performed operation is c = a *. b
     *
     * @param a Pointer to the first Matrix
     * @param b Pointer to the second Matrix
     * @param c Pointer to the result Matrix
     */
    static void componentWiseMult(Matrix *a, Matrix *b, Matrix *c)
    {
        init();
        if(a->getCols() != b->getCols()) return;
        if(a->getRows() != b->getRows()) return;
        if(b->getCols() != c->getCols()) return;
        if(b->getRows() != c->getRows()) return;
        instance->componentWiseMult(a,b,c);
    }


    /**
     * @brief Inplace Matrix Matrix component wise multiplication. The performed operation is a = a *. b
     *
     * @param a Pointer to the first Matrix witch will be overwritten by the result of the operation
     * @param b Pointer to the second Matrix
     */
    static void componentWiseMult(Matrix *a, Matrix *b)
    {
        init();
        if(a->getCols() != b->getCols()) return;
        if(a->getRows() != b->getRows()) return;
        instance->componentWiseMult(a,b);
    }

    /**
     * @brief Inplace 2D fourier transformation
     *
     * @param m
     */
    static void fft(Matrix* m)
    {
        init();
        instance->fft(m);
    }

    /**
     * @brief Inplace inverse 2D fourier transformation
     *
     * @param m
     */
    static void ifft(Matrix* m)
    {
        init();
        instance->ifft(m);
    }

    /**
     * @brief Out of place 2D fourier transformation. NOT IMPLEMENTED JET
     *
     * @param in
     * @param out
     */
    static void fft(Matrix* in, Matrix* out)
    {
        init();
        instance->fft(in, out);
    }

    /**
     * @brief Out of place inverse 2D fourier transformation. NOT IMPLEMENTED JET
     *
     * @param in
     * @param out
     */
    static void ifft(Matrix* in, Matrix* out)
    {
        init();
        instance->ifft(in, out);
    }

    /**
     * @brief Rearranges the outputs of fft by moving the zero-frequency component to the center of the Matrix
     *
     * @param m
     */
    static void fftshift(Matrix* m)
    {
        init();
        instance->fftshift(m);
    }

    /**
     * @brief Reverts the fftshift operation
     *
     * @param m
     */
    static void ifftshift(Matrix* m)
    {
        init();
        instance->ifftshift(m);
    }

    /**
     * @brief Sets the acrive plugin. NOTE: Do not change the plugin at runtime
     *
     * @param plugin
     */
    static void setActivePlugin(MathPlugin* plugin)
    {
        init();
        Q_ASSERT(mathPlugins.contains(plugin));
        instance = plugin;
    }

    /**
     * @brief Returns a list of loaded MathPlugin
     *
     * @return QList<MathPlugin *>
     */
    static QList<MathPlugin*> getAvailablePlugins()
    {
        init();
        return mathPlugins;
    }

    /**
     * @brief Deletes all loaded MathPlugin
     *
     */
    static void destroy()
    {
        foreach(MathPlugin* mp, mathPlugins)
        {
            delete mp;
        }
        initialised = false;
    }
};

#endif // MATH_H
