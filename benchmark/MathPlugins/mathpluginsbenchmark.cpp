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

#include "mathpluginsbenchmark.h"

int isprim(unsigned int z)
{
    unsigned int i;
    for(i = 2; i <= z/2; i++)
    {
        if(!(z % i))
        {
            return 0;
        }
    }
    return 1;
}

QSet<int> primfac(unsigned int z)
{
    QSet<int> r;
    unsigned int i;
    while(z > 1)
    {
        i = 2;
        while(1)
        {
            if(!(z % i) && isprim(i))
            {
                r << i;
                z /= i;
                break;
            }
            i++;
        }
    }
    return r;
}

MathPluginsBenchmark::MathPluginsBenchmark()
{
    // Random init
    QTime midnight(0, 0, 0);
    qsrand(midnight.secsTo(QTime::currentTime()));

    QDir pluginDir = QDir::current();
    pluginDir.cdUp();
    pluginDir.cdUp();
    pluginDir.cdUp();
    pluginDir.cdUp();
    pluginDir.cd("plugins");
    pluginDir.cd("bin");

    foreach (QString fileName, pluginDir.entryList(QDir::Files))
    {
        QPluginLoader pluginLoader(pluginDir.absoluteFilePath(fileName));
        QObject *plugin = pluginLoader.instance();
        if(plugin)
        {
            MathPlugin* mathPlugin = qobject_cast<MathPlugin *>(plugin);
            if(mathPlugin)
            {
                if(mathPlugin->isPlatformSupported())
                {
                    plugins.push_back(mathPlugin);
                }
                else
                {
                    delete mathPlugin;
                }
            }
        }
        else
        {
            qWarning() << pluginLoader.errorString();
        }
    }

}

MathPluginsBenchmark::~MathPluginsBenchmark()
{
    foreach(MathPlugin* p, plugins)
    {
        delete p;
    }
}

void MathPluginsBenchmark::randomMatrixFill(Matrix *m)
{
    for(int i = 0; i < m->getRows(); i++)
    {
        for(int j = 0; j < m->getCols(); j++)
        {
            m->set(i, j, std::complex<double>(RND_DOUBLE, RND_DOUBLE));
        }
    }
}

void MathPluginsBenchmark::randomVectorFill(Vector *m)
{
    for(int i = 0; i < m->getSize(); i++)
    {
        m->set(i, std::complex<double>(RND_DOUBLE, RND_DOUBLE));
    }
}

void MathPluginsBenchmark::matrixVecMult(MathPlugin *plugin, int matrixSize, int operations, bool preSync)
{
    Vector* v = plugin->createVector(matrixSize);
    randomVectorFill(v);
    Matrix* m = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m);
    Vector* r = plugin->createVector(matrixSize);

    if(preSync)
    {
        v->data();
        m->data();
        QBENCHMARK
        {
            for(int i = 0; i < operations; i++)
            {
                plugin->mult(m, v, r);
            }
        }
    }
    else
    {
        QBENCHMARK
        {
            // we set some data to force device mem update on every iteration
            v->set(0,std::complex<double>(1.1,1.1));
            m->set(0,0,std::complex<double>(1.1,1.1));
            for(int i = 0; i < operations; i++)
            {
                plugin->mult(m, v, r);
            }
        }
    }

    delete v;
    delete m;
    delete r;
}

void MathPluginsBenchmark::matrixMatrixMult(MathPlugin *plugin, int matrixSize, int operations, bool preSync)
{
    Matrix* m1 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m1);
    Matrix* m2 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m2);
    Matrix* r = plugin->createMatrix(matrixSize,matrixSize);

    if(preSync)
    {
        m1->data();
        m2->data();
        QBENCHMARK
        {
            for(int i = 0; i < operations; i++)
            {
                plugin->mult(m1, m2, r);
            }
        }
    }
    else
    {
        QBENCHMARK
        {
            // we set some data to force device mem update on every iteration
            m1->set(0,0,std::complex<double>(1.1,1.1));
            m2->set(0,0,std::complex<double>(1.1,1.1));
            for(int i = 0; i < operations; i++)
            {
                plugin->mult(m1, m2, r);
            }
        }
    }

    delete m1;
    delete m2;
    delete r;
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMult(MathPlugin *plugin, int matrixSize, int operations, bool preSync)
{
    Matrix* m1 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m1);
    Matrix* m2 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m2);
    Matrix* r = plugin->createMatrix(matrixSize,matrixSize);

    if(preSync)
    {
        m1->data();
        m2->data();
        QBENCHMARK
        {
            for(int i = 0; i < operations; i++)
            {
                plugin->componentWiseMult(m1, m2, r);
            }
        }
    }
    else
    {
        QBENCHMARK
        {
            // we set some data to force device mem update on every iteration
            m1->set(0,0,std::complex<double>(1.1,1.1));
            m2->set(0,0,std::complex<double>(1.1,1.1));
            for(int i = 0; i < operations; i++)
            {
                plugin->componentWiseMult(m1, m2, r);
            }
        }
    }

    delete m1;
    delete m2;
    delete r;
}

void MathPluginsBenchmark::matrixAdd(MathPlugin *plugin, int matrixSize, int operations, bool preSync)
{
    Matrix* m1 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m1);
    Matrix* m2 = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m2);

    if(preSync)
    {
        m1->data();
        m2->data();
        QBENCHMARK
        {
            for(int i = 0; i < operations; i++)
            {
                plugin->add(m1, m2);
            }
        }
    }
    else
    {
        QBENCHMARK
        {
            // we set some data to force device mem update on every iteration
            m1->set(0,0,std::complex<double>(1.1,1.1));
            m2->set(0,0,std::complex<double>(1.1,1.1));
            for(int i = 0; i < operations; i++)
            {
                plugin->add(m1, m2);
            }
        }
    }

    delete m1;
    delete m2;
}

void MathPluginsBenchmark::matrixFFT(MathPlugin *plugin, int matrixSize, int operations, bool preSync)
{
    Matrix* m = plugin->createMatrix(matrixSize, matrixSize);
    randomMatrixFill(m);

    if(preSync)
    {
        m->data();
        QBENCHMARK
        {
            for(int i = 0; i < operations; i++)
            {
                plugin->fft(m);
            }
        }
    }
    else
    {
        QBENCHMARK
        {
            // we set some data to force device mem update on every iteration
            m->set(0,0,std::complex<double>(1.1,1.1));
            for(int i = 0; i < operations; i++)
            {
                plugin->fft(m);
            }
        }
    }

    delete m;
}

void MathPluginsBenchmark::singlePluginBenchmarkData(QString pluginName)
{
    QTest::addColumn<int>("pluginIndex");
    QTest::addColumn<int>("matrixSize");
    QTest::addColumn<int>("operations");
    QTest::addColumn<bool>("preSync");
    int operations = 1;
    for (int i = 2; i <= 502; i+=10) {
        int pluginIndex = 0;
        foreach(MathPlugin* p, plugins)
        {
            if(p->getName().contains(pluginName))
            {
                QString name = "plugin=[" + p->getName() + "] matrixSize=[" + QString::number(i) + "] operations=[" + QString::number(operations) + "] preSync=[false]";
                QTest::newRow(name.toStdString().c_str()) << pluginIndex << i << operations << false;
            }
            pluginIndex++;
        }
    }
}

void MathPluginsBenchmark::benchmarkData3D()
{
    QTest::addColumn<int>("pluginIndex");
    QTest::addColumn<int>("matrixSize");
    QTest::addColumn<int>("operations");
    QTest::addColumn<bool>("preSync");

    for (int matrixSize = 2; matrixSize <= 502; matrixSize += 50)
    {
        for(int operations = 1; operations <= 101; operations += 10)
        {
            int pluginIndex = 0;
            foreach(MathPlugin* p, plugins)
            {
                if(p->getName().contains("CBlas") ||
                        p->getName().contains("Cuda Math Plugin") ||
                        p->getName().contains("APPML"))
                {
                    QString name = "plugin=[" + p->getName() +
                                   "] matrixSize=[" + QString::number(matrixSize) +
                                   "] operations=[" + QString::number(operations) + "] preSync=[false]";
                    QTest::newRow(name.toStdString().c_str()) << pluginIndex << matrixSize << operations << false;
                }
                pluginIndex++;
            }
        }
    }
}
////////////// Single Plugin matrixVecMult /////////////////////////

void MathPluginsBenchmark::matrixVecMultCBlas_data()
{
    singlePluginBenchmarkData("CBlas");
}

void MathPluginsBenchmark::matrixVecMultCBlas()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixVecMultGSL_data()
{
    singlePluginBenchmarkData("GSL");
}

void MathPluginsBenchmark::matrixVecMultGSL()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixVecMultACML_data()
{
    singlePluginBenchmarkData("ACML");
}

void MathPluginsBenchmark::matrixVecMultACML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixVecMultCuda_data()
{
    singlePluginBenchmarkData("Cuda");
}

void MathPluginsBenchmark::matrixVecMultCuda()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixVecMultAPPML_data()
{
    singlePluginBenchmarkData("APPML");
}

void MathPluginsBenchmark::matrixVecMultAPPML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

////////////// Single Plugin matrixMatrixMult /////////////////////////

void MathPluginsBenchmark::matrixMatrixMultCBlas_data()
{
    singlePluginBenchmarkData("CBlas");
}

void MathPluginsBenchmark::matrixMatrixMultCBlas()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixMultGSL_data()
{
    singlePluginBenchmarkData("GSL");
}

void MathPluginsBenchmark::matrixMatrixMultGSL()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixMultACML_data()
{
    singlePluginBenchmarkData("ACML");
}

void MathPluginsBenchmark::matrixMatrixMultACML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixMultCuda_data()
{
    singlePluginBenchmarkData("Cuda");
}

void MathPluginsBenchmark::matrixMatrixMultCuda()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixMultAPPML_data()
{
    singlePluginBenchmarkData("APPML");
}

void MathPluginsBenchmark::matrixMatrixMultAPPML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

////////////// Single Plugin matrixMatrixComponentWiseMult /////////////////////////

void MathPluginsBenchmark::matrixMatrixComponentWiseMultCBlas_data()
{
    singlePluginBenchmarkData("CBlas");
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultCBlas()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultGSL_data()
{
    singlePluginBenchmarkData("GSL");
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultGSL()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultACML_data()
{
    singlePluginBenchmarkData("ACML");
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultACML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultCuda_data()
{
    singlePluginBenchmarkData("Cuda");
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultCuda()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultAPPML_data()
{
    singlePluginBenchmarkData("APPML");
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMultAPPML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

////////////// Single Plugin matrixAdd /////////////////////////

void MathPluginsBenchmark::matrixAddCBlas_data()
{
    singlePluginBenchmarkData("CBlas");
}

void MathPluginsBenchmark::matrixAddCBlas()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixAdd(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixAddGSL_data()
{
    singlePluginBenchmarkData("GSL");
}

void MathPluginsBenchmark::matrixAddGSL()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixAdd(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixAddACML_data()
{
    singlePluginBenchmarkData("ACML");
}

void MathPluginsBenchmark::matrixAddACML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixAdd(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixAddCuda_data()
{
    singlePluginBenchmarkData("Cuda");
}

void MathPluginsBenchmark::matrixAddCuda()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixAdd(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixAddAPPML_data()
{
    singlePluginBenchmarkData("APPML");
}

void MathPluginsBenchmark::matrixAddAPPML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixAdd(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

////////////// Single Plugin matrixFFT /////////////////////////

void MathPluginsBenchmark::matrixFFTCBlas_data()
{
    singlePluginBenchmarkData("CBlas");
}

void MathPluginsBenchmark::matrixFFTCBlas()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixFFTGSL_data()
{
    singlePluginBenchmarkData("GSL");
}

void MathPluginsBenchmark::matrixFFTGSL()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixFFTACML_data()
{
    singlePluginBenchmarkData("ACML");
}

void MathPluginsBenchmark::matrixFFTACML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixFFTCuda_data()
{
    singlePluginBenchmarkData("Cuda");
}

void MathPluginsBenchmark::matrixFFTCuda()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixFFTAPPML_data()
{
    QTest::addColumn<int>("pluginIndex");
    QTest::addColumn<int>("matrixSize");
    QTest::addColumn<int>("operations");
    QTest::addColumn<bool>("preSync");
    int operations = 1;
    int pluginIndex = 0;
    foreach(MathPlugin* p, plugins)
    {
        if(p->getName().contains("APPML"))
            break;
        pluginIndex++;
    }
    for (int i = 2; i <= 502; i++) {
        // APPML only supports that dimension sizes
        QSet<int> fac = primfac(i);
        fac.remove(2);
        fac.remove(3);
        fac.remove(5);
        if(fac.isEmpty())
        {
            QString name = "plugin=[" + plugins[pluginIndex]->getName() + "] matrixSize=[" + QString::number(i) + "] operations=[" + QString::number(operations) + "] preSync=[false]";
            QTest::newRow(name.toStdString().c_str()) << pluginIndex << i << operations << false;
        }
    }
}

void MathPluginsBenchmark::matrixFFTAPPML()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);
    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

/////////////////////// 3D //////////////////

void MathPluginsBenchmark::matrixVecMult3d_data()
{
    benchmarkData3D();
}

void MathPluginsBenchmark::matrixVecMult3d()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);

    matrixVecMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixMult3d_data()
{
    benchmarkData3D();
}

void MathPluginsBenchmark::matrixMatrixMult3d()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);

    matrixMatrixMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMult3d_data()
{
    benchmarkData3D();
}

void MathPluginsBenchmark::matrixMatrixComponentWiseMult3d()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);

    matrixMatrixComponentWiseMult(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

void MathPluginsBenchmark::matrixFFT3d_data()
{
    benchmarkData3D();
}

void MathPluginsBenchmark::matrixFFT3d()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    QFETCH(int, operations);
    QFETCH(bool, preSync);

    matrixFFT(plugins.at(pluginIndex), matrixSize, operations, preSync);
}

/////////////////////// Other //////////////////

void MathPluginsBenchmark::copyHostToDevice_data()
{
    QTest::addColumn<int>("pluginIndex");
    QTest::addColumn<int>("matrixSize");
    for(int i = 2; i <= 100; i+=2)
    {
        int pluginIndex = 0;
        foreach(MathPlugin* p, plugins)
        {
            if(p->getName().contains("Cuda") ||
                    p->getName().contains("APPML"))
            {
                QString name = "plugin=[" + p->getName() + "] matrixSize=[" + QString::number(i) + "]";
                QTest::newRow(name.toStdString().c_str()) << pluginIndex << i;
            }
            pluginIndex++;
        }
    }
}

void MathPluginsBenchmark::copyHostToDevice()
{
    QFETCH(int, pluginIndex);
    QFETCH(int, matrixSize);
    Matrix* m = plugins.at(pluginIndex)->createMatrix(matrixSize, matrixSize);

    QBENCHMARK
    {
        // we set some data to force device mem update
        m->set(0,0,std::complex<double>(1.1,1.1));
        m->data();
    }
    delete m;


}
QTEST_APPLESS_MAIN(MathPluginsBenchmark)
