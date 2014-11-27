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

#ifndef HIPNOSCOMPONENTTEST_H
#define HIPNOSCOMPONENTTEST_H

#include <QObject>
#include <QtTest/QtTest>
#include <complex>

#include "common/math/matrix.h"
#include "common/pipelinedata.h"

#define EPSILON 0.0001

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class HipnosComponentTest : public QObject
{
    Q_OBJECT

public:
/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
    HipnosComponentTest();

protected:
    /**
     * @brief
     *
     * @param n
     */
    /**
     * @brief
     *
     * @param n
     */
    QString complexToQString(const std::complex<double> &n);
    /**
     * @brief
     *
     * @param expected
     * @param got
     */
    /**
     * @brief
     *
     * @param expected
     * @param got
     */
    QString error(double expected, double got);
    /**
     * @brief
     *
     * @param expected
     * @param got
     */
    /**
     * @brief
     *
     * @param expected
     * @param got
     */
    QString error(std::complex<double> expected, std::complex<double> got);
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
    bool compare(double a, double b);
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
    bool compare(std::complex<double> a, std::complex<double> b);
    /**
     * @brief
     *
     * @param m
     * @param fileName
     */
    /**
     * @brief
     *
     * @param m
     * @param fileName
     */
    void compareMatrixWithCsv(Matrix* m, QString fileName);
    /**
     * @brief
     *
     * @param g
     * @param f
     */
    /**
     * @brief
     *
     * @param g
     * @param f
     */
    float compareGaussWithFourierData(GaussPipelineData* g, FourierPipelineData* f);
    /**
     * @brief
     *
     * @param g
     * @param f
     */
    /**
     * @brief
     *
     * @param g
     * @param f
     */
    float compareGaussWithFourierIntesity(GaussPipelineData* g, FourierPipelineData* f);

};

#endif // HIPNOSCOMPONENTTEST_H
