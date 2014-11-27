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

#include "hipnoscomponenttest.h"
#include <cmath>

HipnosComponentTest::HipnosComponentTest()
{
}

QString HipnosComponentTest::error(double expected, double got)
{
    return "Expected: " + QString::number(expected) + ", Got: " + QString::number(got);
}

QString HipnosComponentTest::error(std::complex<double> expected, std::complex<double> got)
{
    return "Expected: " + complexToQString(expected) + ", Got: " + complexToQString(got);
}

QString HipnosComponentTest::complexToQString(const std::complex<double> &n)
{
    return "(" + QString::number(n.real()) + " + " + QString::number(n.imag()) + "i)";
}

bool HipnosComponentTest::compare(double a, double b)
{
    int expa;
    double mana = std::frexp(a, &expa);
    int expb;
    double manb = std::frexp(b, &expb);
    if(expa != expb)
        return false;
    return std::fabs(mana - manb) < EPSILON;
}


bool HipnosComponentTest::compare(std::complex<double> a, std::complex<double> b)
{
    return compare(std::real(a), std::real(b)) && compare(std::imag(a), std::imag(b));
}

void HipnosComponentTest::compareMatrixWithCsv(Matrix *m, QString fileName)
{
    QFile file(fileName);
    double biggestDiff = 0;
    std::complex<double> biggestDiffVal1, biggestDiffVal2;
    int biggestDiffRow, biggestDiffCol;

    if (!file.open(QIODevice::ReadOnly))
    {
        qFatal("Error opening file " + fileName.toLocal8Bit());
    }

    int row = 0;
    while (!file.atEnd()){
        QString line = file.readLine();
        int col = 0;
        foreach(QString numString, line.split(","))
        {
            std::complex<double> csvData;
            if(numString == "0" || numString == "-0")
            {
                csvData = std::complex<double>(0, 0);
            }
            else if(numString == "1")
            {
                csvData = std::complex<double>(1, 0);
            }
            else
            {
                // parse complex number
                int sep = numString.indexOf(QRegExp("[^e][-+]"))+1;
                QString left = numString.left(sep);
                QString right = numString.right(numString.length() - sep).remove("i");
                csvData = std::complex<double>(left.toDouble(), right.toDouble());
            }
            double abs = std::abs(csvData - m->get(row,col));
            if(abs > biggestDiff)
            {
                biggestDiff = abs;
                biggestDiffVal1 = csvData;
                biggestDiffVal2 = m->get(row,col);
                biggestDiffRow = row;
                biggestDiffCol = col;
            }

            QVERIFY2(compare(csvData, m->get(row,col)),
                     ("At (" + QString::number(row) + "," + QString::number(col) + ") " + error(csvData, m->get(row,col))).toAscii());

            col++;
            if(col > m->getCols())
            {
                qDebug() << "CSV has more cols than the matrix";
                return;
            }
        }
        row++;
        if(row > m->getRows())
        {
            qDebug() << "CSV has more rows than the matrix";
            return;
        }
    }
    file.close();
    qDebug() << "Biggest diff for file" << fileName << "was found in row" << biggestDiffRow
             << "and col" << biggestDiffCol << ".";
    qDebug() << "csv:\t" << complexToQString(biggestDiffVal1).toStdString().c_str();
    qDebug() << "calc:\t" << complexToQString(biggestDiffVal2).toStdString().c_str();
    qDebug() << "---------------------------" ;
    qDebug() << "abs:\t" << biggestDiff;
}

float HipnosComponentTest::compareGaussWithFourierData(GaussPipelineData *g, FourierPipelineData *f)
{
    Q_ASSERT(g->Resolution == f->Resolution);

    Matrix* gm = g->getComplexAmplitude();
    Matrix* fm = f->getComplexAmplitude();

    float biggestDiff = 0;
    std::complex<double> biggestDiffVal1, biggestDiffVal2;
    int biggestDiffRow, biggestDiffCol;

    for(int x = 0; x < g->Resolution; x++)
    {
        for(int y = 0; y < g->Resolution; y++)
        {
            double abs = std::abs(fm->get(x,y) - gm->get(x,y));
            if(abs > biggestDiff)
            {
                biggestDiff = abs;
                biggestDiffVal1 = fm->get(x,y);
                biggestDiffVal2 =gm->get(x,y);
                biggestDiffRow = x;
                biggestDiffCol = y;
            }
        }
    }

    qDebug() << "Biggest diff was found in row" << biggestDiffRow
             << "and col" << biggestDiffCol << ".";
    qDebug() << "Fourier:\t" << complexToQString(biggestDiffVal1).toStdString().c_str();
    qDebug() << "Gauss:\t" << complexToQString(biggestDiffVal2).toStdString().c_str();
    qDebug() << "---------------------------" ;
    qDebug() << "abs:\t" << biggestDiff;

    delete gm;
    delete fm;

    return biggestDiff;
}

float HipnosComponentTest::compareGaussWithFourierIntesity(GaussPipelineData *g, FourierPipelineData *f)
{
    Q_ASSERT(g->Resolution == f->Resolution);

    Matrix* gm = g->getComplexAmplitude();
    Matrix* fm = f->getComplexAmplitude();

    double biggestDiff = 0;
    double biggestDiffVal1, biggestDiffVal2;
    int biggestDiffRow, biggestDiffCol;

    for(int x = 0; x < g->Resolution; x++)
    {
        for(int y = 0; y < g->Resolution; y++)
        {
            double fd = std::abs(fm->get(x,y));
            fd *= fd;
            double gd = std::abs(gm->get(x,y));
            gd *= gd;

            //float abs = std::abs(fd - gd)/(fd + gd);
            float abs = std::abs(fd - gd);
            if(abs > biggestDiff)
            {
                biggestDiff = abs;
                biggestDiffVal1 = fd;
                biggestDiffVal2 =gd;
                biggestDiffRow = x;
                biggestDiffCol = y;
            }
        }
    }

    qDebug() << "Biggest diff was found in row" << biggestDiffRow
             << "and col" << biggestDiffCol << ".";
    qDebug() << "Fourier:\t" << biggestDiffVal1;
    qDebug() << "Gauss:\t" << biggestDiffVal2;
    qDebug() << "---------------------------" ;
    qDebug() << "abs:\t" << biggestDiff << "";

    delete fm;
    delete gm;

    return biggestDiff;
}
