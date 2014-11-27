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

#include "propagationcomponent.h"

PropagationComponent::PropagationComponent(double l, PropagationMethod pm) :
    PipelineComponent()
{
    length = l;
    propagationMethod = pm;
    setNumberOfInputConnections(1);
    setNumberOfOutputConnections(1);
}

PropagationComponent::~PropagationComponent()
{
}

QString PropagationComponent::getType()
{
    return "Propagation";
}

QIcon PropagationComponent::getIcon()
{
    return QIcon(":/icons/components/drift.png");
}

PipelineComponent* PropagationComponent::clone()
{
    return new PropagationComponent(length);
}

void PropagationComponent::setProperty(Property p)
{
    if(p.getName() == "Length" && p.getType() == Property::PT_DOUBLE)
    {
        length = p.getDoubleValue();
    }
    if(p.getName() == "Propagation Method" && p.getType() == Property::PT_LIST)
    {
        if(p.getSelectedValue() == "Near Field")
            propagationMethod = PM_NEAR_FIELD;
        else
            propagationMethod = PM_FAR_FIELD;
    }
}

QList<PipelineComponent::Property> PropagationComponent::getProperties()
{
    QList<Property> l;
    l << Property(this, "Length", "Specifies the length of the free space propagation", length, 0.0, 100000.0, "m");

    QList<QString> options;
    options << "Near Field" << "Far Field";
    if(propagationMethod == PM_NEAR_FIELD)
        l << Property(this, "Propagation Method", "Specifies the propagation method", "Near Field", options);
    else
        l << Property(this, "Propagation Method", "Specifies the propagation method", "Far Field", options);
    return l;
}

void PropagationComponent::gaussPropagation(double z)
{    
    PipelineSpectrumData* spectrum = getInputConnection()->getData(DT_GAUSS);
    for(int i = 0; i < spectrum->getSize(); i++)
    {
        GaussPipelineData* data = spectrum->getData<GaussPipelineData>(i);
        if(z > length)
            z = length;
        data->q += z;
    }
    getOutputConnection()->setData(spectrum);
}

void PropagationComponent::fourierPropagation(double z)
{
    PipelineSpectrumData* spectrum = getInputConnection()->getData(DT_FOURIER);
    for(int i = 0; i < spectrum->getSize(); i++)
    {
        FourierPipelineData* data = spectrum->getData<FourierPipelineData>(i);

        if(propagationMethod == PM_NEAR_FIELD)
        {
            double freq = 1.0 / data->SamplingStepSize;
            Matrix* fx = Math::createMatrix(data->Resolution, data->Resolution);
            Matrix* fy = Math::createMatrix(data->Resolution, data->Resolution);
            Math::forMatrices(fx, fy, freq/data->Resolution, freq/data->Resolution);
            Math::pow(fx, 2);
            Math::pow(fy, 2);
            Math::add(fx, fy);
            delete fy;
            Math::mult(fx, std::complex<double>(0, M_PI * z * data->Lambda));
            Math::exp(fx);
            Math::mult(fx, std::exp(std::complex<double>(0, -2.0 * M_PI * z / data->Lambda)));

            Math::ifftshift(data->ComplexAmplitude);
            Math::fft(data->ComplexAmplitude);

            Math::ifftshift(fx);
            Math::componentWiseMult(data->ComplexAmplitude, fx);
            delete fx;

            Math::ifft(data->ComplexAmplitude);
            Math::fftshift(data->ComplexAmplitude);
        }
        else if(propagationMethod == PM_FAR_FIELD)
        {
            if(z > 0)
            {
                Math::ifftshift(data->ComplexAmplitude);
                Math::fft(data->ComplexAmplitude);
                Math::fftshift(data->ComplexAmplitude);
                std::complex<double> h0 = 1.0/std::complex<double>(0, (data->Lambda*z)) *
                        std::exp(std::complex<double>(0,-2 * M_PI * z / data->Lambda));
                Math::mult(data->ComplexAmplitude, h0 * data->SamplingStepSize * data->SamplingStepSize);
                data->SamplingStepSize = data->Lambda * z / (data->SamplingStepSize*(double)data->Resolution);
            }
        }
    }

    getOutputConnection()->setData(spectrum);
}
