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

#include "pipelinedata.h"

/******************************************************************
  PipelineData
 ******************************************************************/

PipelineData::PipelineData(double lambda, int resolution, double stepSize)
{
    Lambda = lambda;
    Resolution = resolution;
    SamplingStepSize = stepSize;
}

PipelineData::~PipelineData()
{
}

PipelineDataType PipelineData::getDataType()
{
    return dataType;
}

/******************************************************************
  GaussPipelineData
 ******************************************************************/

GaussPipelineData::GaussPipelineData(double lambda, double w, double r,  double e0, int resolution)
    : PipelineData(lambda, resolution, w * std::sqrt(M_PI*resolution) / resolution)
{
    dataType = DT_GAUSS;
    E0 = e0;

    calculateWaistPositionAndSize(w, r);

    if(r == 0)
    {
        q = std::complex<double>(0,Z0);
    }
    else
    {
        q = std::complex<double>(1.0,0.0) / std::complex<double>(1.0/r , -Lambda/( M_PI * w * w ));
    }
}

GaussPipelineData::GaussPipelineData(const GaussPipelineData& copy)
    : PipelineData(copy.Lambda, copy.Resolution, copy.SamplingStepSize)
{
    dataType = DT_GAUSS;
    E0 = copy.E0;
    WaistPosition = copy.WaistPosition;
    W0 = copy.W0;
    Z0 = copy.Z0;
    q = copy.q;
}

void GaussPipelineData::calculateWaistPositionAndSize(double w, double r)
{
    if(r == 0)
    {
        WaistPosition = 0;
        W0 = w;
    }
    else
    {
        double tmp = (Lambda*r)/(M_PI*w*w);
        WaistPosition = r / ( 1.0 + tmp*tmp);
        W0 = w / (std::sqrt( 1.0 + 1.0/(tmp*tmp)));
    }
    Z0 = M_PI * W0 * W0 / Lambda;
    //qDebug() << "WaistPosition" << WaistPosition << "W0" << W0;
}

GaussPipelineData::~GaussPipelineData()
{
}

PipelineData* GaussPipelineData::clone()
{
    GaussPipelineData* clone = new GaussPipelineData(*this);
    return clone;
}

double GaussPipelineData::getZ()
{
    double w = getW();
    if(w < W0)
        w = W0; // assume w == W0 probably a rounding error
    return Z0 * std::sqrt(((w*w)/(W0*W0))-1.0);
}

double GaussPipelineData::getW()
{
    // imag(1/q) = - lambda /(pi * w^2)
    // w^2 = -lambda / (pi *imag(1/q))
    double imagOfqInvers = -std::imag(std::complex<double>(1,0)/q);
    return std::sqrt(Lambda/(M_PI*imagOfqInvers));
}

double GaussPipelineData::getR()
{
    return 1.0/ std::real(std::complex<double>(1,0)/q);
}

double GaussPipelineData::getI()
{
    double i0 = E0*E0;
    double w0FractWz = W0 / getW();
    return i0 * w0FractWz * w0FractWz;
}

Matrix* GaussPipelineData::getComplexAmplitude()
{
    double z = getZ();
    double gouyPhase = std::atan(z/Z0);
    double k = 2 * M_PI / Lambda;
    std::complex<double> Eq = E0 * (W0/getW()) * std::exp(
                std::complex<double>(0, -k*z) + // -ikz
                std::complex<double>(0,gouyPhase)
                );

    Matrix* fx = Math::createMatrix(Resolution, Resolution);
    Matrix* fy = Math::createMatrix(Resolution, Resolution);
    Math::forMatrices(fx, fy, SamplingStepSize, SamplingStepSize);
    Math::pow(fx, 2);
    Math::pow(fy, 2);
    Math::add(fx, fy);
    delete fy;
    Math::mult(fx, std::complex<double>(0,-k/2.0) / q); //-ik/(2q)
    Math::exp(fx);
    Math::mult(fx, Eq);
    return fx;
}

/******************************************************************
  FourierPipelineData
 ******************************************************************/

FourierPipelineData::FourierPipelineData(double lambda, int resolution, double stepSize)
    : PipelineData(lambda, resolution, stepSize)
{
    ComplexAmplitude = Math::createMatrix(resolution, resolution);
    dataType = DT_FOURIER;
}

FourierPipelineData::FourierPipelineData(const FourierPipelineData& copy)
    : PipelineData(copy.Lambda, copy.Resolution, copy.SamplingStepSize)
{
    dataType = DT_FOURIER;
    ComplexAmplitude = copy.ComplexAmplitude->clone();
}

FourierPipelineData::~FourierPipelineData()
{
    delete ComplexAmplitude;
}

PipelineData* FourierPipelineData::clone()
{
    FourierPipelineData* clone = new FourierPipelineData(*this);
    return clone;
}

Matrix* FourierPipelineData::getComplexAmplitude()
{
    return ComplexAmplitude->clone();
}

/******************************************************************
  PipelineSpectrumData
 ******************************************************************/

PipelineSpectrumData* PipelineSpectrumData::createGaussSpectrumData(Spectrum s, double w, double r, double e, int res)
{
    PipelineSpectrumData* spectrumData = new PipelineSpectrumData(s, DT_GAUSS);

    double spectrumNorm = s.getNormalizationFactor();

    for(int i = 0; i < s.size(); i++)
    {
        spectrumData->setData(i, new GaussPipelineData(1.0/s.getEntry(i).Frequency, w, r, e * s.getEntry(i).Intensity * s.getStepWidth(i) / spectrumNorm, res));
    }
    return spectrumData;
}

PipelineSpectrumData* PipelineSpectrumData::createFourierSpectrumData(Spectrum s, int res, double stepSize)
{
    PipelineSpectrumData* spectrumData = new PipelineSpectrumData(s, DT_FOURIER);
    for(int i = 0; i < s.size(); i++)
    {
        spectrumData->setData(i, new FourierPipelineData(1.0/s.getEntry(i).Frequency, res, stepSize));
    }
    return spectrumData;
}

PipelineSpectrumData::PipelineSpectrumData(Spectrum s, PipelineDataType dt){
    spectrum = s;
    dataType = dt;
    data.resize(spectrum.size());
}

PipelineSpectrumData::~PipelineSpectrumData(){
    foreach(PipelineData* d, data)
    {
        delete d;
    }
}

int PipelineSpectrumData::getSize()
{
    return spectrum.size();
}

PipelineDataType PipelineSpectrumData::getDataType()
{
    return dataType;
}

PipelineSpectrumData* PipelineSpectrumData::clone()
{
    PipelineSpectrumData* clone = new PipelineSpectrumData(spectrum, dataType);
    for(int i = 0; i < spectrum.size(); i++)
    {
        clone->setData(i, data[i]->clone());
    }
    return clone;
}

template<class T> T* PipelineSpectrumData::getData(int i)
{
    qFatal("This template function can only be called with FourierPipelineData or GaussPipelineData as template class");
}

template<> FourierPipelineData* PipelineSpectrumData::getData<>(int i)
{
    Q_ASSERT(getDataType() == DT_FOURIER);
    Q_ASSERT(data[i]->getDataType() == DT_FOURIER);
    return static_cast<FourierPipelineData*>(data[i]);
}

template<> GaussPipelineData* PipelineSpectrumData::getData<>(int i)
{
    Q_ASSERT(getDataType() == DT_GAUSS);
    Q_ASSERT(data[i]->getDataType() == DT_GAUSS);
    return static_cast<GaussPipelineData*>(data[i]);
}

template<> PipelineData* PipelineSpectrumData::getData<>(int i)
{
    return data[i];
}

PipelineData* PipelineSpectrumData::getData(int i)
{
    return data[i];
}

void PipelineSpectrumData::setData(int i, PipelineData *d)
{
    data[i] = d;
}

Matrix* PipelineSpectrumData::getComplexAmplitude(double fromFrequency, double toFrequency)
{
    if(data.size() == 0)
    {
        return 0;
    }
    else
    {
        Matrix* result = 0;
        for(int i = 0; i < data.size(); i++)
        {
            if(spectrum.getEntry(i).Frequency >= fromFrequency &&
                    spectrum.getEntry(i).Frequency <= toFrequency)
            {
                if(result == 0)
                    result = data[i]->getComplexAmplitude();
                else
                {
                    Matrix* tmp = data[i]->getComplexAmplitude();
                    Math::add(result, tmp);
                    delete tmp;
                }
            }
        }
        return result;
    }
}

Matrix* PipelineSpectrumData::getComplexAmplitude()
{
    if(data.size() == 0)
    {
        return 0;
    }
    else
    {
        Matrix* result = data[0]->getComplexAmplitude();
        for(int i = 1; i < data.size(); i++)
        {
            Matrix* tmp = data[i]->getComplexAmplitude();
            Math::add(result, tmp);
            delete tmp;
        }
        return result;
    }
}

double PipelineSpectrumData::getSamplingStepSize()
{
    if(data.size() == 0)
    {
        return 0;
    }
    else
    {
        double result = data[0]->SamplingStepSize;
        for(int i = 1; i < data.size(); i++)
        {
            if(result > data[i]->SamplingStepSize)
                result = data[i]->SamplingStepSize;
        }
        return result;
    }
}

Spectrum PipelineSpectrumData::getSpectrum()
{
    return spectrum;
}


double PipelineSpectrumData::getSpectrumNormalizationFactor()
{
    return spectrum.getNormalizationFactor();
}
