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

#include "gaussopticstest.h"
#include "common/components/gaussianbeamsourcecomponent.h"
#include "common/components/propagationcomponent.h"
#include "common/components/thinlenscomponent.h"

GaussOpticsTest::GaussOpticsTest()
{
}

void GaussOpticsTest::testPropagation()
{
    double lambda = 1030 * 0.000000001; //m
    double W0 = 0.002; //m
    double E0 = 1;
    int resolution = 256;
    int phaseradius = 0;

    GaussianBeamSourceComponent source(lambda, W0, phaseradius, E0, resolution);
    PropagationComponent drift(100, PropagationComponent::PM_NEAR_FIELD);

    PipelineConnection conn1(&source, &drift);
    source.setOutputConnection(&conn1);
    drift.setInputConnection(&conn1);

    PipelineSink sink(&drift);
    drift.setOutputConnection(&sink);

    // Test Gauss
    qDebug() << "Testing SourceComponent ...";
    PipelineSpectrumData* gaussSpectrumData = conn1.getData(DT_GAUSS);
    GaussPipelineData* gaussData = gaussSpectrumData->getData<GaussPipelineData>(0);
    PipelineSpectrumData* fourierSpectrumData = conn1.getData(DT_FOURIER);
    FourierPipelineData* fourierData = fourierSpectrumData->getData<FourierPipelineData>(0);
    float maxAbs = compareGaussWithFourierData(gaussData, fourierData);
    qDebug() << "Error: " << (maxAbs / E0 * 100) << "%";
    delete gaussSpectrumData;
    delete fourierSpectrumData;

    // Test Propagation
    QFile file("propagation-error.data");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;
    qDebug() << "Testing PropagationComponent ...";
    for(int i = 0; i <= 100; i+=1)
    {
        gaussSpectrumData = sink.fetchIntermediatDataFromInput(DT_GAUSS, i);
        gaussData = gaussSpectrumData->getData<GaussPipelineData>(0);
        fourierSpectrumData = sink.fetchIntermediatDataFromInput(DT_FOURIER, i);
        fourierData = fourierSpectrumData->getData<FourierPipelineData>(0);
        maxAbs = compareGaussWithFourierIntesity(gaussData, fourierData);
        qDebug() << "Error: " << QString::number(maxAbs / E0 * 100) << "%";
        file.write((QString::number(i) + "\t" + QString::number(maxAbs) + "\n").toLocal8Bit());
        delete fourierSpectrumData;
        delete gaussSpectrumData;
    }
    file.close();
}

void GaussOpticsTest::testLens()
{
    double lambda = 1030 * 0.000000001; //m
    double W0 = 0.002; //m
    double E0 = 1;
    int resolution = 256;
    int phaseradius = 0;

    GaussianBeamSourceComponent source(lambda, W0, phaseradius, E0, resolution);
    ThinLensComponent lens(25, 0, 0);
    PropagationComponent drift(100, PropagationComponent::PM_NEAR_FIELD);

    PipelineConnection conn1(&source, &lens);
    source.setOutputConnection(&conn1);
    lens.setInputConnection(&conn1);

    PipelineConnection conn2(&lens, &drift);
    lens.setOutputConnection(&conn2);
    drift.setInputConnection(&conn2);

    PipelineSink sink(&drift);
    drift.setOutputConnection(&sink);

    // Test Gauss
    qDebug() << "Testing SourceComponent ...";
    PipelineSpectrumData* gaussSpectrumData =  conn1.getData(DT_GAUSS);
    GaussPipelineData* gaussData = gaussSpectrumData->getData<GaussPipelineData>(0);
    PipelineSpectrumData* fourierSpectrumData =  conn1.getData(DT_FOURIER);
    FourierPipelineData* fourierData = fourierSpectrumData->getData<FourierPipelineData>(0);
    float maxAbs = compareGaussWithFourierData(gaussData, fourierData);
    qDebug() << "Error: " << (maxAbs / E0 * 100) << "%";
    delete gaussSpectrumData;
    delete fourierSpectrumData;

    // Test Lens
    QFile lensFile("lens-error.data");
    if (!lensFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;
    qDebug() << "Testing ThinLensComponent ...";
    gaussSpectrumData = conn2.getData(DT_GAUSS);
    gaussData = gaussSpectrumData->getData<GaussPipelineData>(0);
    fourierSpectrumData = conn2.getData(DT_FOURIER);
    fourierData = fourierSpectrumData->getData<FourierPipelineData>(0);

    Matrix* gausMatrix = gaussData->getComplexAmplitude();
    for(int x = 0; x < gaussData->Resolution; x++)
    {
        for(int y = 0; y < gaussData->Resolution; y++)
        {
            double ga = std::arg(gausMatrix->get(x,y));
            double fa = std::arg(fourierData->ComplexAmplitude->get(x,y));
            double diff = ga - fa;
            lensFile.write((QString::number(x) + "\t" + QString::number(y) + "\t" + QString::number(ga) + "\t" +
                             QString::number(fa) + "\t" + QString::number(diff) + "\n").toLocal8Bit());
        }
        lensFile.write("\n");
    }

    delete gausMatrix;
    delete gaussSpectrumData;
    delete fourierSpectrumData;
    lensFile.close();

    // Test Propagation
    QFile propFile("propagation-with-lens-error.data");
    if (!propFile.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return;
    qDebug() << "Testing PropagationComponent ...";
    for(int i = 0; i < 100; i+=1)
    {
        gaussSpectrumData = conn2.fetchIntermediatDataFromInput(DT_GAUSS, i);
        gaussData = gaussSpectrumData->getData<GaussPipelineData>(0);
        fourierSpectrumData = conn2.fetchIntermediatDataFromInput(DT_FOURIER, i);
        fourierData = fourierSpectrumData->getData<FourierPipelineData>(0);
        maxAbs = compareGaussWithFourierIntesity(gaussData, fourierData);
        qDebug() << "Error: " << QString::number(maxAbs / E0 * 100) << "%";
        propFile.write((QString::number(i) + "\t" + QString::number(maxAbs) + "\n").toLocal8Bit());
        delete gaussSpectrumData;
        delete fourierSpectrumData;
    }
    propFile.close();
}
