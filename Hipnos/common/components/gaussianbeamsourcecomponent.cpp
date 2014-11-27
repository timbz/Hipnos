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

#include "gaussianbeamsourcecomponent.h"
#include "common/math/math.h"
#include <QDebug>

GaussianBeamSourceComponent::GaussianBeamSourceComponent(Spectrum s, double w, double r, double e, int res) :
    PipelineComponent()
{
    setNumberOfInputConnections(0);
    setNumberOfOutputConnections(1);
    w0 = w;
    phaseradius = r;
    e0 = e;
    resolution = res;
    spectrum = s;
}

GaussianBeamSourceComponent::~GaussianBeamSourceComponent()
{
}

QString GaussianBeamSourceComponent::getType()
{
    return "Gaussian beam source";
}

QIcon GaussianBeamSourceComponent::getIcon()
{
    return QIcon(":/icons/components/source.png");
}

PipelineComponent* GaussianBeamSourceComponent::clone()
{
    return new GaussianBeamSourceComponent(spectrum, w0, phaseradius, e0, resolution);
}

void GaussianBeamSourceComponent::setProperty(Property p)
{
    if(p.getName() == "W0" && p.getType() == Property::PT_DOUBLE)
    {
        w0 = p.getDoubleValue();
    }
    else if(p.getName() == "Phaseradius" && p.getType() == Property::PT_DOUBLE)
    {
        phaseradius = p.getDoubleValue();
    }
    else if(p.getName() == "E0" && p.getType() == Property::PT_DOUBLE)
    {
        e0 = p.getDoubleValue();
    }
    else if(p.getName() == "Resolution" && p.getType() == Property::PT_INT)
    {
        resolution = p.getIntValue();
    }
    else if(p.getName() == "Spectrum" && p.getType() == Property::PT_SPECTRUM)
    {
        spectrum = p.getSpectrum();
    }
}

QList<PipelineComponent::Property> GaussianBeamSourceComponent::getProperties()
{
    QList<Property> l;
    l << Property(this, "W0", "Specifies the waist size",
                  w0, 0.0,  100000.0, "m");
    l << Property(this, "Phaseradius", "Specifies the initial radius of curvature of the wavefronts. A value of zero corresponds to infinity (no curvature)",
                  phaseradius, -100000.0,  100000.0, "m");
    l << Property(this, "E0", "Specifies the  electric field amplitude" ,
                  e0, 0.0,  100000.0, "V/m");
    l << Property(this, "Spectrum", "Sets the spectrum", spectrum);
    Property res(this, "Resolution", "Specifies the resolution of the sampled data. Only even numbers are allowed", resolution, 16,  2048, 2);
    res.setValidator(&resolutionValidator);
    l << res;
    return l;
}

void GaussianBeamSourceComponent::gaussPropagation(double z)
{
    getOutputConnection()->setData(
                PipelineSpectrumData::createGaussSpectrumData(spectrum, w0, phaseradius, e0, resolution));
}

void GaussianBeamSourceComponent::fourierPropagation(double z)
{
    PipelineSpectrumData* spectrumData =
            PipelineSpectrumData::createFourierSpectrumData(spectrum, resolution, w0 * std::sqrt(M_PI*resolution) / resolution);

    double spectrumNorm = spectrumData->getSpectrumNormalizationFactor();

    for(int i = 0; i < spectrumData->getSize(); i++)
    {
        FourierPipelineData* data = spectrumData->getData<FourierPipelineData>(i);

        // TODO: set as param
    //    double centerx = 0; //m
    //    double centery = 0; //m
        Matrix* fy = Math::createMatrix(resolution, resolution);
        Math::forMatrices(data->ComplexAmplitude, fy, data->SamplingStepSize, data->SamplingStepSize);
        Math::pow(data->ComplexAmplitude, 2);
        Math::pow(fy, 2);
        Math::add(data->ComplexAmplitude, fy);
        delete fy;
        if(phaseradius == 0)
        {
            Math::mult(data->ComplexAmplitude, std::complex<double>(-1/(w0*w0), 0));
        }
        else
        {
            Math::mult(data->ComplexAmplitude, std::complex<double>(-1/(w0*w0), -M_PI/(phaseradius*data->Lambda)));
        }
        Math::exp(data->ComplexAmplitude);
        // scale by e0, spectrum.I and spectrumNorm
        Math::mult(data->ComplexAmplitude, std::complex<double>(e0 * spectrum.getEntry(i).Intensity / spectrumNorm, 0));
    }

    getOutputConnection()->setData(spectrumData);
}
