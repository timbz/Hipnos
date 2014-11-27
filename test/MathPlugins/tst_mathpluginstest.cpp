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

#include <QtCore/QString>
#include <QtTest/QtTest>

#include "common/math/mathplugin.h"

class MathPluginsTest : public QObject
{
    Q_OBJECT

private:
    QList<MathPlugin*> plugins;
    QString complexToQString(std::complex<double> n);
    QString errorMessage(MathPlugin* math, std::complex<double> exp, std::complex<double> got);
    void printMatrix(Matrix* m);

public:
    MathPluginsTest();
    ~MathPluginsTest();

private Q_SLOTS:
    void testMatrixClone();
    void testVectorClone();
    void testMatrixMatrixMult();
    void testMatrixVectorMult();
    void testMatrixMatrixComponentWiseMult();
    void testMatrixFft();
    void testMatrixFftshift();
    void testMatrixFftAndFftshift();
    void testMatrixAdd();
    void testMatrixExp();
    void testMatrixPow();
    void testMatrixScalarMult();
    void testForMatrices();
};

MathPluginsTest::MathPluginsTest()
{
    qDebug() << "Loading Math Plugins ...";
    QDir pluginDir = QDir::current();
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
                    qDebug() << mathPlugin->getName() << "loaded.";
                    plugins.push_back(mathPlugin);
                }
                else
                {
                    qDebug() << mathPlugin->getName() << "not supported.";
                    delete mathPlugin;
                }
            }
        }
        else
        {
            qWarning() << pluginLoader.errorString();
        }
    }
    qDebug() << "Found" << plugins.size() << "Math Plugins";
}

MathPluginsTest::~MathPluginsTest()
{
    foreach(MathPlugin* p, plugins)
    {
        delete p;
    }
}

QString MathPluginsTest::complexToQString(std::complex<double> n)
{
    return "(" + QString::number(n.real()) + " + " + QString::number(n.imag()) + "i)";
}

QString MathPluginsTest::errorMessage(MathPlugin* math, std::complex<double> exp, std::complex<double> got)
{
    return math->getName() + ": expected " + complexToQString(exp) + ", got " + complexToQString(got);
}

void MathPluginsTest::printMatrix(Matrix *m)
{
    for(int i = 0; i < m->getRows(); i++)
    {
        QString row = "";
        for(int j = 0; j < m->getCols(); j++)
        {
            row += QString::number(m->get(i,j).real()) + "," + QString::number(m->get(i,j).imag()) + "\t";
        }
        qDebug() << row;
    }
}

void MathPluginsTest::testMatrixMatrixMult()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,3);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(0,2,3);
        a->set(1,0,4);
        a->set(1,1,5);
        a->set(1,2,6);

        Matrix* b = math->createMatrix(3,2);
        b->set(0,0,7);
        b->set(0,1,8);
        b->set(1,0,9);
        b->set(1,1,10);
        b->set(2,0,11);
        b->set(2,1,12);

        Matrix* c = math->createMatrix(2,2);
        math->mult(a,b,c);

        QVERIFY2(c->get(0,0) == std::complex<double>(58), errorMessage(math, std::complex<double>(58), c->get(0,0)).toAscii());
        QVERIFY2(c->get(0,1) == std::complex<double>(64), errorMessage(math, std::complex<double>(64), c->get(0,1)).toAscii());
        QVERIFY2(c->get(1,0) == std::complex<double>(139), errorMessage(math, std::complex<double>(139), c->get(1,0)).toAscii());
        QVERIFY2(c->get(1,1) == std::complex<double>(154), errorMessage(math, std::complex<double>(154), c->get(1,1)).toAscii());

        delete a;
        delete b;
        delete c;
    }
}

void MathPluginsTest::testMatrixClone()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        Matrix* b = a->clone();

        QVERIFY2(b->get(0,0) == a->get(0,0), errorMessage(math, a->get(0,0), b->get(0,0)).toAscii());
        QVERIFY2(b->get(0,1) == a->get(0,1), errorMessage(math, a->get(0,1), b->get(0,1)).toAscii());
        QVERIFY2(b->get(1,0) == a->get(1,0), errorMessage(math, a->get(1,0), b->get(1,0)).toAscii());
        QVERIFY2(b->get(1,1) == a->get(1,1), errorMessage(math, a->get(1,1), b->get(1,1)).toAscii());

        delete a;
        delete b;
    }
}

void MathPluginsTest::testVectorClone()
{
    foreach(MathPlugin* math, plugins)
    {
        Vector* a = math->createVector(2);
        a->set(0, 9);
        a->set(1, 10);

        Vector* b = a->clone();

        QVERIFY2(b->get(0) == a->get(0), errorMessage(math, a->get(0), b->get(0)).toAscii());
        QVERIFY2(b->get(1) == a->get(1), errorMessage(math, a->get(1), b->get(1)).toAscii());

        delete a;
        delete b;
    }
}

void MathPluginsTest::testMatrixVectorMult()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        Vector* e = math->createVector(2);
        e->set(0, 9);
        e->set(1, 10);

        Vector* f = math->createVector(2);
        math->mult(a, e, f);

        QVERIFY2(f->get(0) == std::complex<double>(29), errorMessage(math, std::complex<double>(29), f->get(0)).toAscii());
        QVERIFY2(f->get(1) == std::complex<double>(67), errorMessage(math, std::complex<double>(67), f->get(1)).toAscii());

        delete a;
        delete e;
        delete f;
    }
}

void MathPluginsTest::testMatrixMatrixComponentWiseMult()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,3);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(0,2,3);
        a->set(1,0,4);
        a->set(1,1,5);
        a->set(1,2,6);

        Matrix* b = math->createMatrix(2,3);
        b->set(0,0,7);
        b->set(0,1,8);
        b->set(0,2,9);
        b->set(1,0,10);
        b->set(1,1,11);
        b->set(1,2,12);

        Matrix* c = math->createMatrix(2,3);
        math->componentWiseMult(a,b,c);
        QVERIFY2(c->get(0,0) == std::complex<double>(7), errorMessage(math, std::complex<double>(7), c->get(0,0)).toAscii());
        QVERIFY2(c->get(0,1) == std::complex<double>(16), errorMessage(math, std::complex<double>(16), c->get(0,1)).toAscii());
        QVERIFY2(c->get(0,2) == std::complex<double>(27), errorMessage(math, std::complex<double>(27), c->get(0,2)).toAscii());
        QVERIFY2(c->get(1,0) == std::complex<double>(40), errorMessage(math, std::complex<double>(40), c->get(1,0)).toAscii());
        QVERIFY2(c->get(1,1) == std::complex<double>(55), errorMessage(math, std::complex<double>(55), c->get(1,1)).toAscii());
        QVERIFY2(c->get(1,2) == std::complex<double>(72), errorMessage(math, std::complex<double>(72), c->get(1,2)).toAscii());

        math->componentWiseMult(a,b);
        QVERIFY2(a->get(0,0) == std::complex<double>(7), errorMessage(math, std::complex<double>(7), a->get(0,0)).toAscii());
        QVERIFY2(a->get(0,1) == std::complex<double>(16), errorMessage(math, std::complex<double>(16), a->get(0,1)).toAscii());
        QVERIFY2(a->get(0,2) == std::complex<double>(27), errorMessage(math, std::complex<double>(27), a->get(0,2)).toAscii());
        QVERIFY2(a->get(1,0) == std::complex<double>(40), errorMessage(math, std::complex<double>(40), a->get(1,0)).toAscii());
        QVERIFY2(a->get(1,1) == std::complex<double>(55), errorMessage(math, std::complex<double>(55), a->get(1,1)).toAscii());
        QVERIFY2(a->get(1,2) == std::complex<double>(72), errorMessage(math, std::complex<double>(72), a->get(1,2)).toAscii());

        delete a;
        delete b;
        delete c;
    }
}

void MathPluginsTest::testMatrixFft()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,4);
        a->set(0,0,std::complex<double>(1,1));
        a->set(0,1,std::complex<double>(2,2));
        a->set(0,2,std::complex<double>(3,3));
        a->set(0,3,std::complex<double>(4,4));
        a->set(1,0,std::complex<double>(5,5));
        a->set(1,1,std::complex<double>(6,6));
        a->set(1,2,std::complex<double>(7,7));
        a->set(1,3,std::complex<double>(8,8));

        Matrix* b = a->clone();
        math->fft(b);
        math->ifft(b);

        double epsilon = 1e-14;
        QVERIFY2(std::abs(a->get(0,0) - b->get(0,0)) < epsilon, errorMessage(math, a->get(0,0), b->get(0,0)).toAscii());
        QVERIFY2(std::abs(a->get(0,1) - b->get(0,1)) < epsilon, errorMessage(math, a->get(0,1), b->get(0,1)).toAscii());
        QVERIFY2(std::abs(a->get(0,2) - b->get(0,2)) < epsilon, errorMessage(math, a->get(0,2), b->get(0,2)).toAscii());
        QVERIFY2(std::abs(a->get(0,3) - b->get(0,3)) < epsilon, errorMessage(math, a->get(0,3), b->get(0,3)).toAscii());
        QVERIFY2(std::abs(a->get(1,0) - b->get(1,0)) < epsilon, errorMessage(math, a->get(1,0), b->get(1,0)).toAscii());
        QVERIFY2(std::abs(a->get(1,1) - b->get(1,1)) < epsilon, errorMessage(math, a->get(1,1), b->get(1,1)).toAscii());
        QVERIFY2(std::abs(a->get(1,2) - b->get(1,2)) < epsilon, errorMessage(math, a->get(1,2), b->get(1,2)).toAscii());
        QVERIFY2(std::abs(a->get(1,3) - b->get(1,3)) < epsilon, errorMessage(math, a->get(1,3), b->get(1,3)).toAscii());

        delete a;
        delete b;
    }
}

void MathPluginsTest::testMatrixFftshift()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,4);
        a->set(0,0,std::complex<double>(1,1));
        a->set(0,1,std::complex<double>(2,2));
        a->set(0,2,std::complex<double>(3,3));
        a->set(0,3,std::complex<double>(4,4));
        a->set(1,0,std::complex<double>(5,5));
        a->set(1,1,std::complex<double>(6,6));
        a->set(1,2,std::complex<double>(7,7));
        a->set(1,3,std::complex<double>(8,8));

        math->fftshift(a);

        QVERIFY2(a->get(0,0) == std::complex<double>(7,7), errorMessage(math, std::complex<double>(7,7), a->get(0,0)).toAscii());
        QVERIFY2(a->get(0,1) == std::complex<double>(8,8), errorMessage(math, std::complex<double>(8,8), a->get(0,1)).toAscii());
        QVERIFY2(a->get(0,2) == std::complex<double>(5,5), errorMessage(math, std::complex<double>(5,5), a->get(0,2)).toAscii());
        QVERIFY2(a->get(0,3) == std::complex<double>(6,6), errorMessage(math, std::complex<double>(6,6), a->get(0,3)).toAscii());
        QVERIFY2(a->get(1,0) == std::complex<double>(3,3), errorMessage(math, std::complex<double>(3,3), a->get(1,0)).toAscii());
        QVERIFY2(a->get(1,1) == std::complex<double>(4,4), errorMessage(math, std::complex<double>(4,4), a->get(1,1)).toAscii());
        QVERIFY2(a->get(1,2) == std::complex<double>(1,1), errorMessage(math, std::complex<double>(1,1), a->get(1,2)).toAscii());
        QVERIFY2(a->get(1,3) == std::complex<double>(2,2), errorMessage(math, std::complex<double>(2,2), a->get(1,3)).toAscii());

        delete a;
    }
}


void MathPluginsTest::testMatrixFftAndFftshift()
{
    double epsilon = 1e-14;

    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,4);
        a->set(0,0,std::complex<double>(1,1));
        a->set(0,1,std::complex<double>(2,2));
        a->set(0,2,std::complex<double>(3,3));
        a->set(0,3,std::complex<double>(4,4));
        a->set(1,0,std::complex<double>(5,5));
        a->set(1,1,std::complex<double>(6,6));
        a->set(1,2,std::complex<double>(7,7));
        a->set(1,3,std::complex<double>(8,8));

        Matrix* b = a->clone();
        math->ifftshift(b);
        math->fft(b);
        math->ifft(b);
        math->fftshift(b);

        QVERIFY2(std::abs(a->get(0,0) - b->get(0,0)) < epsilon, errorMessage(math, a->get(0,0), b->get(0,0)).toAscii());
        QVERIFY2(std::abs(a->get(0,1) - b->get(0,1)) < epsilon, errorMessage(math, a->get(0,1), b->get(0,1)).toAscii());
        QVERIFY2(std::abs(a->get(0,2) - b->get(0,2)) < epsilon, errorMessage(math, a->get(0,2), b->get(0,2)).toAscii());
        QVERIFY2(std::abs(a->get(0,3) - b->get(0,3)) < epsilon, errorMessage(math, a->get(0,3), b->get(0,3)).toAscii());
        QVERIFY2(std::abs(a->get(1,0) - b->get(1,0)) < epsilon, errorMessage(math, a->get(1,0), b->get(1,0)).toAscii());
        QVERIFY2(std::abs(a->get(1,1) - b->get(1,1)) < epsilon, errorMessage(math, a->get(1,1), b->get(1,1)).toAscii());
        QVERIFY2(std::abs(a->get(1,2) - b->get(1,2)) < epsilon, errorMessage(math, a->get(1,2), b->get(1,2)).toAscii());
        QVERIFY2(std::abs(a->get(1,3) - b->get(1,3)) < epsilon, errorMessage(math, a->get(1,3), b->get(1,3)).toAscii());

        delete a;
        delete b;
    }
}

void MathPluginsTest::testMatrixAdd()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        Matrix* b = math->createMatrix(2,2);
        b->set(0,0,5);
        b->set(0,1,6);
        b->set(1,0,7);
        b->set(1,1,8);

        math->add(a, b);
        QVERIFY2(a->get(0, 0) == std::complex<double>(6), errorMessage(math, std::complex<double>(6), a->get(0,0)).toAscii());
        QVERIFY2(a->get(0, 1) == std::complex<double>(8), errorMessage(math, std::complex<double>(8), a->get(0,1)).toAscii());
        QVERIFY2(a->get(1, 0) == std::complex<double>(10), errorMessage(math, std::complex<double>(10), a->get(1,0)).toAscii());
        QVERIFY2(a->get(1, 1) == std::complex<double>(12), errorMessage(math, std::complex<double>(12), a->get(1,1)).toAscii());

        delete a;
        delete b;
    }
}

void MathPluginsTest::testMatrixPow()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        math->pow(a, 2);
        QVERIFY2(a->get(0, 0) == std::complex<double>(1), errorMessage(math, std::complex<double>(1), a->get(0,0)).toAscii());
        QVERIFY2(a->get(0, 1) == std::complex<double>(4), errorMessage(math, std::complex<double>(4), a->get(0,1)).toAscii());
        QVERIFY2(a->get(1, 0) == std::complex<double>(9), errorMessage(math, std::complex<double>(9), a->get(1,0)).toAscii());
        QVERIFY2(a->get(1, 1) == std::complex<double>(16), errorMessage(math, std::complex<double>(16), a->get(1,1)).toAscii());

        delete a;
    }
}

void MathPluginsTest::testMatrixScalarMult()
{
    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        math->mult(a, 2);
        QVERIFY2(a->get(0, 0) == std::complex<double>(2), errorMessage(math, std::complex<double>(2), a->get(0,0)).toAscii());
        QVERIFY2(a->get(0, 1) == std::complex<double>(4), errorMessage(math, std::complex<double>(4), a->get(0,1)).toAscii());
        QVERIFY2(a->get(1, 0) == std::complex<double>(6), errorMessage(math, std::complex<double>(6), a->get(1,0)).toAscii());
        QVERIFY2(a->get(1, 1) == std::complex<double>(8), errorMessage(math, std::complex<double>(8), a->get(1,1)).toAscii());

        delete a;
    }
}

void MathPluginsTest::testMatrixExp()
{
    double epsilon = 1e-5;

    foreach(MathPlugin* math, plugins)
    {
        Matrix* a = math->createMatrix(2,2);
        a->set(0,0,1);
        a->set(0,1,2);
        a->set(1,0,3);
        a->set(1,1,4);

        math->exp(a);
        QVERIFY2(std::abs(a->get(0, 0) - std::exp(std::complex<double>(1))) < epsilon, errorMessage(math, std::exp(std::complex<double>(1)), a->get(0,0)).toAscii());
        QVERIFY2(std::abs(a->get(0, 1) - std::exp(std::complex<double>(2))) < epsilon, errorMessage(math, std::exp(std::complex<double>(2)), a->get(0,1)).toAscii());
        QVERIFY2(std::abs(a->get(1, 0) - std::exp(std::complex<double>(3))) < epsilon, errorMessage(math, std::exp(std::complex<double>(3)), a->get(1,0)).toAscii());
        QVERIFY2(std::abs(a->get(1, 1) - std::exp(std::complex<double>(4))) < epsilon, errorMessage(math, std::exp(std::complex<double>(4)), a->get(1,1)).toAscii());

        delete a;
    }
}

void MathPluginsTest::testForMatrices()
{
    double stepx = 0.5;
    double stepy = 1.0;
    foreach(MathPlugin* math, plugins)
    {
        Matrix* fx = math->createMatrix(4,3);
        Matrix* fy = math->createMatrix(4,3);
        math->forMatrices(fx, fy, stepx, stepy);
//        qDebug() << math->getName();
//        printMatrix(fx);
//        printMatrix(fy);
        for(int i = 0; i < fx->getRows(); i++)
        {
            std::complex<double> y(stepy * (double)(i - fx->getRows()/2), 0);
            for(int j = 0; j < fx->getCols(); j++)
            {
                std::complex<double> x(stepx * (double)(j - fx->getCols()/2), 0);
                QVERIFY2(fx->get(i, j) == x ,errorMessage(math, x, fx->get(i,j)).toAscii());
                QVERIFY2(fy->get(i, j) == y ,errorMessage(math, y, fy->get(i,j)).toAscii());
            }
        }
        delete fx;
        delete fy;
    }
}

QTEST_APPLESS_MAIN(MathPluginsTest)

#include "tst_mathpluginstest.moc"
