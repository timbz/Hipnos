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

#ifndef PIPELINEDATA_H
#define PIPELINEDATA_H

#define _USE_MATH_DEFINES
#include <math.h>

#include <complex>
#include "math/math.h"
#include "spectrum.h"

/**
 * @brief Pipeline data types that define the underlying physical computation model
 *
 * Types:
 * - DT_GAUSS used by the gaussian beam computation model
 * - DT_FOURIER used by the fourier optics computation model
 */
enum PipelineDataType
{
    DT_GAUSS,
    DT_FOURIER
};

/**
 * @brief Base class for all pipeline data implementations
 *
 */
struct PipelineData
{
    double Lambda;   /**< The wavelength */
    int Resolution;   /**< The resolution of the data set */
    double SamplingStepSize;   /**< The sampling step of data set */

    PipelineData(double lambda, int resolution, double stepSize);
    virtual ~PipelineData();
    PipelineDataType getDataType();

    virtual PipelineData* clone() = 0;
    virtual Matrix* getComplexAmplitude() = 0;

protected: // set this to protected to prevent from writing to it
    PipelineDataType dataType;  
};

/**
 * @brief Implementation of a PipelineData used for the gaussian beam computation model
 *
 */
struct GaussPipelineData : public PipelineData
{
    double Z0;   /**< Rayleigh range */
    double W0;   /**< Waist radius */
    double E0;   /**< Inital electric field amplitude */
    double WaistPosition;   /**< Waist position */
    std::complex<double> q;   /**< Complex beam parameter */

    GaussPipelineData(double lambda, double w, double r,  double e0, int resolution);
    ~GaussPipelineData();
    PipelineData* clone();
    Matrix* getComplexAmplitude();
    void calculateWaistPositionAndSize(double w, double r);
    double getW();
    double getI();
    double getZ();
    double getR();

private:
    GaussPipelineData(const GaussPipelineData &copy); // used by clone
};

/**
 * @brief Implementation of a PipelineData used for the fourier optics computation model
 *
 */
struct FourierPipelineData : public PipelineData
{
    Matrix* ComplexAmplitude;  /**< A pointer to a Matrix containing the complex amplitude of the beam */

    FourierPipelineData(double lambda, int resolution, double stepSize);
    ~FourierPipelineData();

    PipelineData* clone();
    Matrix* getComplexAmplitude();

private:
    FourierPipelineData(const FourierPipelineData &copy); // used by clone
};

/**
 * @brief A array of PipelineData used to propagate a Spectrum trough the Pipeline
 *
 */
class PipelineSpectrumData
{

public:
    static PipelineSpectrumData* createGaussSpectrumData(Spectrum s, double w, double r, double e, int res);
    static PipelineSpectrumData* createFourierSpectrumData(Spectrum s, int res, double stepSize);

    ~PipelineSpectrumData();

    template<class T> T* getData(int i);
    PipelineData* getData(int i);
    void setData(int i, PipelineData* d);
    int getSize();
    PipelineDataType getDataType();
    PipelineSpectrumData* clone();
    Matrix* getComplexAmplitude(double fromFrequency, double toFrequency);
    Matrix* getComplexAmplitude();
    double getSamplingStepSize();
    Spectrum getSpectrum();
    double getSpectrumNormalizationFactor();

private:
    PipelineSpectrumData(Spectrum s, PipelineDataType dt);
    PipelineDataType dataType;  
    Spectrum spectrum;  
    QVector<PipelineData*> data;  
};

#endif // PIPELINEDATA_H
