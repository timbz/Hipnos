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

#include "fourieropticstest.h"
#include "common/pipelinedata.h"
#include "common/components/gaussianbeamsourcecomponent.h"
#include "common/components/aperturecomponent.h"
#include "common/components/propagationcomponent.h"
#include "common/components/thinlenscomponent.h"

FourierOpticsTest::FourierOpticsTest()
{
}

void FourierOpticsTest::testFourier()
{
    double lambda = 1030 * 0.000000001; //m
    double W0 = 0.002; //m
    double E0 = 1;
    int resolution = 256;
    int phaseradius = 10;

    GaussianBeamSourceComponent source(Spectrum(lambda), W0, phaseradius, E0, resolution);
    CircularApertureComponent aperture(0.004, 0.004, 0, 0, 0);
    PropagationComponent drift(100, PropagationComponent::PM_NEAR_FIELD);
    ThinLensComponent lens(25, 0, 0);
    PropagationComponent drift2(100, PropagationComponent::PM_FAR_FIELD);

    PipelineConnection conn1(&source, &aperture);
    source.setOutputConnection(&conn1);
    aperture.setInputConnection(&conn1);

    PipelineConnection conn2(&aperture, &drift);
    aperture.setOutputConnection(&conn2);
    drift.setInputConnection(&conn2);

    PipelineConnection conn3(&drift, &lens);
    drift.setOutputConnection(&conn3);
    lens.setInputConnection(&conn3);

    PipelineConnection conn4(&lens, &drift2);
    lens.setOutputConnection(&conn4);
    drift2.setInputConnection(&conn4);

    PipelineSink sink(&drift2);
    drift2.setOutputConnection(&sink);

    // Test Gauss
    PipelineSpectrumData* spectrum = conn1.getData(DT_FOURIER);
    FourierPipelineData* data = spectrum->getData<FourierPipelineData>(0);
    qDebug() << "Testing SourceComponent output...";
    compareMatrixWithCsv(data->ComplexAmplitude, ":/TestData/gaussfill.csv");
    delete spectrum;

    // Test Aperture
    spectrum = conn2.getData(DT_FOURIER);
    data = spectrum->getData<FourierPipelineData>(0);
    qDebug() << "Testing ApertureComponent transmittance matrix...";
    compareMatrixWithCsv(aperture.getFourierTransmittanceMatrix(), ":/TestData/aperture-trans.csv");
    qDebug() << "Testing ApertureComponent output...";
    compareMatrixWithCsv(data->ComplexAmplitude, ":/TestData/aperture.csv");
    delete spectrum;

    // Test Fresnel
    spectrum = conn3.getData(DT_FOURIER);
    data = spectrum->getData<FourierPipelineData>(0);
    qDebug() << "Testing PropagationComponent(Fresnel) output...";
    compareMatrixWithCsv(data->ComplexAmplitude, ":/TestData/propagation.csv");
    delete spectrum;

    // Test Lense
    spectrum = conn4.getData(DT_FOURIER);
    data = spectrum->getData<FourierPipelineData>(0);
    qDebug() << "Testing ThinsLenseComponent output...";
    compareMatrixWithCsv(data->ComplexAmplitude, ":/TestData/lens.csv");
    delete spectrum;

    // Test Frauenhofer
    spectrum = sink.getData(DT_FOURIER);
    data = spectrum->getData<FourierPipelineData>(0);
    qDebug() << "Testing PropagationComponent(Frauenhofer) output...";
    compareMatrixWithCsv(data->ComplexAmplitude, ":/TestData/propagation2.csv");
    delete spectrum;
}
