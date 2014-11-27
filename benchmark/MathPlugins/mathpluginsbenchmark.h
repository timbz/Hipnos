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

#ifndef MATHPLUGINSBENCHMARK_H
#define MATHPLUGINSBENCHMARK_H

#include <QtCore/QString>
#include <QtTest/QtTest>

#include "common/math/mathplugin.h"

#define RND_DOUBLE (double)qrand()/RAND_MAX

/**
 * @brief
 *
 */
/**
 * @brief
 *
 */
class MathPluginsBenchmark : public QObject
{
    Q_OBJECT

private:
    QList<MathPlugin*> plugins;  

    void randomMatrixFill(Matrix* m);
    void randomVectorFill(Vector* m);

    void singlePluginBenchmarkData(QString pluginName);

    void benchmarkData3D();

    void matrixVecMult(MathPlugin* plugin, int matrixSize, int operations, bool preSync);
    
    void matrixMatrixMult(MathPlugin* plugin, int matrixSize, int operations, bool preSync);
    
    void matrixAdd(MathPlugin* plugin, int matrixSize, int operations, bool preSync);

    void matrixMatrixComponentWiseMult(MathPlugin* plugin, int matrixSize, int operations, bool preSync);

    void matrixFFT(MathPlugin* plugin, int matrixSize, int operations, bool preSync);

public:

    MathPluginsBenchmark();

    ~MathPluginsBenchmark();

private Q_SLOTS:

    // Single Plugins benchmarks

    // --- Vector = Matrix x Vector ---

    void matrixVecMultCBlas_data();

    void matrixVecMultCBlas();


    void matrixVecMultGSL_data();

    void matrixVecMultGSL();


    void matrixVecMultACML_data();

    void matrixVecMultACML();


    void matrixVecMultCuda_data();

    void matrixVecMultCuda();


    void matrixVecMultAPPML_data();

    void matrixVecMultAPPML();

    // --- Matrix = Matrix x Matrix ---

    void matrixMatrixMultCBlas_data();

    void matrixMatrixMultCBlas();


    void matrixMatrixMultGSL_data();

    void matrixMatrixMultGSL();


    void matrixMatrixMultACML_data();

    void matrixMatrixMultACML();


    void matrixMatrixMultCuda_data();

    void matrixMatrixMultCuda();


    void matrixMatrixMultAPPML_data();

    void matrixMatrixMultAPPML();

    // --- Matrix = Matrix .x Matrix ---

    void matrixMatrixComponentWiseMultCBlas_data();

    void matrixMatrixComponentWiseMultCBlas();


    void matrixMatrixComponentWiseMultGSL_data();

    void matrixMatrixComponentWiseMultGSL();


    void matrixMatrixComponentWiseMultACML_data();

    void matrixMatrixComponentWiseMultACML();


    void matrixMatrixComponentWiseMultCuda_data();

    void matrixMatrixComponentWiseMultCuda();


    void matrixMatrixComponentWiseMultAPPML_data();

    void matrixMatrixComponentWiseMultAPPML();

    // --- Matrix = Matrix + Matrix ---

    void matrixAddCBlas_data();

    void matrixAddCBlas();


    void matrixAddGSL_data();

    void matrixAddGSL();


    void matrixAddACML_data();

    void matrixAddACML();


    void matrixAddCuda_data();

    void matrixAddCuda();


    void matrixAddAPPML_data();

    void matrixAddAPPML();

    // --- FFT Matrix ---

    void matrixFFTCBlas_data();

    void matrixFFTCBlas();


    void matrixFFTGSL_data();

    void matrixFFTGSL();


    void matrixFFTACML_data();

    void matrixFFTACML();


    void matrixFFTCuda_data();

    void matrixFFTCuda();


    void matrixFFTAPPML_data();

    void matrixFFTAPPML();

    // --- 3D ---

    void matrixVecMult3d_data();

    void matrixVecMult3d();


    void matrixMatrixMult3d_data();

    void matrixMatrixMult3d();


    void matrixMatrixComponentWiseMult3d_data();

    void matrixMatrixComponentWiseMult3d();


    void matrixFFT3d_data();

    void matrixFFT3d();

    // --- Other ---

    void copyHostToDevice_data();

    void copyHostToDevice();
};

#endif // MATHPLUGINSBENCHMARK_H
