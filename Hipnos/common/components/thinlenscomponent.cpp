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

#include "thinlenscomponent.h"
#include <QDebug>

ThinLensComponent::ThinLensComponent(double f, double x, double y) :
    PipelineComponent()
{
    focalLength = f;
    xCenter = x;
    yCenter = y;
    setNumberOfInputConnections(1);
    setNumberOfOutputConnections(1);
}

ThinLensComponent::~ThinLensComponent()
{
}

QString ThinLensComponent::getType()
{
    return "Thin lens";
}

QIcon ThinLensComponent::getIcon()
{
    return QIcon(":/icons/components/thinlens.png");
}

PipelineComponent* ThinLensComponent::clone()
{
    return new ThinLensComponent(focalLength);
}

void ThinLensComponent::setProperty(Property p)
{
    if(p.getName() == "Focal length" && p.getType() == Property::PT_DOUBLE)
    {
        focalLength = p.getDoubleValue();
    }
    if(p.getName() == "Center X" && p.getType() == Property::PT_DOUBLE)
    {
        xCenter = p.getDoubleValue();
    }
    if(p.getName() == "Center Y" && p.getType() == Property::PT_DOUBLE)
    {
        yCenter = p.getDoubleValue();
    }
}

QList<PipelineComponent::Property> ThinLensComponent::getProperties()
{
    QList<Property> l;
    l << Property(this, "Focal length", "Specifies the focal length of the thin lens",
                  focalLength, -100000.0, 100000.0, "m");
    l << Property(this, "Center X", "Specifies the x position of the lens",
                  xCenter, -100000.0, 100000.0, "m");
    l << Property(this, "Center Y", "Specifies the y position of the lens",
                  yCenter, -100000.0, 100000.0, "m");
    return l;
}

void ThinLensComponent::gaussPropagation(double z)
{
    PipelineSpectrumData* spectrum = getInputConnection()->getData(DT_GAUSS);
    for(int i = 0; i < spectrum->getSize(); i++)
    {
        GaussPipelineData* data = spectrum->getData<GaussPipelineData>(i);
        data->q = data->q / (-data->q/focalLength + std::complex<double>(1,0));

        // update waist position, size and scale E0 according to the new W0
        double oldW0 = data->W0;
        data->calculateWaistPositionAndSize(data->getW(), data->getR());
        data->E0 = data->E0 * oldW0/data->W0;
    }
    getOutputConnection()->setData(spectrum);
}

void ThinLensComponent::fourierPropagation(double z)
{
    PipelineSpectrumData* spectrum = getInputConnection()->getData(DT_FOURIER);
    for(int i = 0; i < spectrum->getSize(); i++)
    {
        FourierPipelineData* data = spectrum->getData<FourierPipelineData>(i);

        Matrix* fourierTransmittanceMatrix = Math::createMatrix(data->Resolution, data->Resolution);

        Matrix* fy = Math::createMatrix(data->Resolution, data->Resolution);
        Math::forMatrices(fourierTransmittanceMatrix, fy, data->SamplingStepSize, data->SamplingStepSize);

        Math::pow(fourierTransmittanceMatrix, 2);
        Math::pow(fy, 2);
        Math::add(fourierTransmittanceMatrix, fy);
        delete fy;
        Math::mult(fourierTransmittanceMatrix, std::complex<double>(0, M_PI / (focalLength*data->Lambda)));
        Math::exp(fourierTransmittanceMatrix);

        Math::componentWiseMult(data->ComplexAmplitude, fourierTransmittanceMatrix);

        delete fourierTransmittanceMatrix;
    }
    getOutputConnection()->setData(spectrum);
}
