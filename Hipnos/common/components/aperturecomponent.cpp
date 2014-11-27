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

#include "aperturecomponent.h"

AbstractApertureComponent::AbstractApertureComponent(double xw, double yw, double xc, double yc, double a) :
    PipelineComponent()
{
    fourierTransmittanceMatrix = 0;
    fourierTransmittanceMatrixSamplingStep = 0;
    width = xw;
    height = yw;
    xCenter = xc;
    yCenter = yc;
    angle = a;
    setNumberOfInputConnections(1);
    setNumberOfOutputConnections(1);
}

AbstractApertureComponent::~AbstractApertureComponent()
{
    if(fourierTransmittanceMatrix)
        delete fourierTransmittanceMatrix;
}

void AbstractApertureComponent::setProperty(Property p)
{
    if(p.getName() == "Center X" && p.getType() == Property::PT_DOUBLE)
        xCenter = p.getDoubleValue();
    if(p.getName() == "Center Y" && p.getType() == Property::PT_DOUBLE)
        yCenter = p.getDoubleValue();
    if(p.getName() == "Width" && p.getType() == Property::PT_DOUBLE)
        width = p.getDoubleValue();
    if(p.getName() == "Height" && p.getType() == Property::PT_DOUBLE)
        height = p.getDoubleValue();
}

QList<PipelineComponent::Property> AbstractApertureComponent::getProperties()
{
    QList<Property> l;
    l << Property(this, "Center X", "Specifies the position of the aperture", xCenter, -10000.0, 10000.0, "m");
    l << Property(this, "Center Y", "Specifies the position of the aperture", yCenter, -10000.0, 10000.0, "m");
    l << Property(this, "Width", "Specifies the width of the aperture", width, 0, 10000.0, "m");
    l << Property(this, "Height", "Specifies the height of the aperture", height, 0, 10000.0, "m");
    return l;
}

void AbstractApertureComponent::gaussPropagation(double z)
{
    getOutputConnection()->setData(getInputConnection()->getData(DT_GAUSS));
}

Matrix* AbstractApertureComponent::getFourierTransmittanceMatrix()
{
    return fourierTransmittanceMatrix;
}

void AbstractApertureComponent::fourierPropagation(double z)
{
    PipelineSpectrumData* spectrum = getInputConnection()->getData(DT_FOURIER);

    for(int i = 0; i < spectrum->getSize(); i++)
    {
        FourierPipelineData* data = spectrum->getData<FourierPipelineData>(i);
        if(changed || fourierTransmittanceMatrix == 0 ||
                fourierTransmittanceMatrix->getCols() != data->ComplexAmplitude->getCols() ||
                fourierTransmittanceMatrix->getRows() != data->ComplexAmplitude->getRows() ||
                fourierTransmittanceMatrixSamplingStep != data->SamplingStepSize)
        {
            updateFourierTransmittanceMatrix(data->Resolution, data->SamplingStepSize);
        }

        Math::componentWiseMult(data->ComplexAmplitude, fourierTransmittanceMatrix);
    }
    getOutputConnection()->setData(spectrum);
}

//------------- CircularApertureComponent -------------

QString CircularApertureComponent::getType()
{
    return "Circular aperture";
}

PipelineComponent* CircularApertureComponent::clone()
{
    return new CircularApertureComponent(width, height, xCenter, yCenter, angle);
}

QIcon CircularApertureComponent::getIcon()
{
    return QIcon(":/icons/components/circularaperture.png");
}

void CircularApertureComponent::updateFourierTransmittanceMatrix(int resolution, double samplingStepSize)
{
    if(fourierTransmittanceMatrix)
        delete fourierTransmittanceMatrix;
    fourierTransmittanceMatrix = Math::createMatrix(resolution, resolution);
    fourierTransmittanceMatrixSamplingStep = samplingStepSize;

    // offset to center at 0,0
    double offest = samplingStepSize * resolution/2;
    // precompute some stuff before the loop
    double xWidthSquare = width*width/4;
    double yWidthSquare = height*height/4;
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    for(int i = 0; i < resolution; i++)
    {
        double yvalue = samplingStepSize * i + yCenter - offest;
        for(int j = 0; j < resolution; j++)
        {
            double xvalue = samplingStepSize * j - xCenter - offest;
            // (cos(an)*(j-yelements/2-yc-0.5)+sin(an)*(k-xelements/2-xc-0.5)).^2./rdy^2+
            // (-sin(an)*(j-yelements/2-yc-0.5)+cos(an)*(k-xelements/2-xc-0.5)).^2./rdx^2 < 1;
            double tmp1 = cosAngle*yvalue + sinAngle*xvalue;
            double tmp2 = -sinAngle*yvalue + cosAngle*xvalue;
            fourierTransmittanceMatrix->set(i, j,
                                    ((tmp1*tmp1)/yWidthSquare +
                                    (tmp2*tmp2)/xWidthSquare) < 1
            );
        }
    }
}

//------------- CircularObstacleComponent -------------

QString CircularObstacleComponent::getType()
{
    return "Circular obstacle";
}

PipelineComponent* CircularObstacleComponent::clone()
{
    return new CircularObstacleComponent(width, height, xCenter, yCenter, angle);
}

QIcon CircularObstacleComponent::getIcon()
{
    return QIcon(":/icons/components/circularobstacle.png");
}

void CircularObstacleComponent::updateFourierTransmittanceMatrix(int resolution, double samplingStepSize)
{
    if(fourierTransmittanceMatrix)
        delete fourierTransmittanceMatrix;
    fourierTransmittanceMatrix = Math::createMatrix(resolution, resolution);

    // offset to center at 0,0
    double offest = samplingStepSize * resolution/2;
    // precompute some stuff before the loop
    double xWidthSquare = width*width/4;
    double yWidthSquare = height*height/4;
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    for(int i = 0; i < resolution; i++)
    {
        double yvalue = samplingStepSize * i + yCenter - offest;
        for(int j = 0; j < resolution; j++)
        {
            double xvalue = samplingStepSize * j - xCenter - offest;
            // (cos(an)*(j-yelements/2-yc-0.5)+sin(an)*(k-xelements/2-xc-0.5)).^2./rdy^2+
            // (-sin(an)*(j-yelements/2-yc-0.5)+cos(an)*(k-xelements/2-xc-0.5)).^2./rdx^2 < 1;
            double tmp1 = cosAngle*yvalue + sinAngle*xvalue;
            double tmp2 = -sinAngle*yvalue + cosAngle*xvalue;
            fourierTransmittanceMatrix->set(i, j,
                                    ((tmp1*tmp1)/yWidthSquare +
                                    (tmp2*tmp2)/xWidthSquare) >= 1
            );
        }
    }
}

//------------- RectangularApertureComponent -------------

QString RectangularApertureComponent::getType()
{
    return "Rectangular aperture";
}

PipelineComponent* RectangularApertureComponent::clone()
{
    return new RectangularApertureComponent(width, height, xCenter, yCenter, angle);
}

QIcon RectangularApertureComponent::getIcon()
{
    return QIcon(":/icons/components/rectangularaperture.png");
}

void RectangularApertureComponent::updateFourierTransmittanceMatrix(int resolution, double samplingStepSize)
{
    if(fourierTransmittanceMatrix)
        delete fourierTransmittanceMatrix;
    fourierTransmittanceMatrix = Math::createMatrix(resolution, resolution);
    fourierTransmittanceMatrixSamplingStep = samplingStepSize;

    // offset to center at 0,0
    double offest = samplingStepSize * resolution/2;
    // precompute some stuff before the loop
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    for(int i = 0; i < resolution; i++)
    {
        double yvalue = samplingStepSize * i + yCenter - offest;
        for(int j = 0; j < resolution; j++)
        {
            double xvalue = samplingStepSize * j - xCenter - offest;
            // ( (abs(cos(an)*(j-yelements/2-yc-0.5)+sin(an)*(k-xelements/2-xc-0.5))>wdy) |
            // (abs(-sin(an)*(j-yelements/2-yc-0.5)+cos(an)*(k-xelements/2-xc-0.5))>wdx));
            double tmp1 = cosAngle*yvalue + sinAngle*xvalue;
            double tmp2 = -sinAngle*yvalue + cosAngle*xvalue;
            fourierTransmittanceMatrix->set(i, j, (std::abs(tmp1) < width/2 & std::abs(tmp2) < height/2));
        }
    }
}


//------------- RectangularObstacleComponent -------------

QString RectangularObstacleComponent::getType()
{
    return "Rectangular obstacle";
}

PipelineComponent* RectangularObstacleComponent::clone()
{
    return new RectangularObstacleComponent(width, height, xCenter, yCenter, angle);
}

QIcon RectangularObstacleComponent::getIcon()
{
    return QIcon(":/icons/components/rectangularobstacle.png");
}

void RectangularObstacleComponent::updateFourierTransmittanceMatrix(int resolution, double samplingStepSize)
{
    if(fourierTransmittanceMatrix)
        delete fourierTransmittanceMatrix;
    fourierTransmittanceMatrix = Math::createMatrix(resolution, resolution);
    fourierTransmittanceMatrixSamplingStep = samplingStepSize;

    // offset to center at 0,0
    double offest = samplingStepSize * resolution/2;
    // precompute some stuff before the loop
    double cosAngle = std::cos(angle);
    double sinAngle = std::sin(angle);
    for(int i = 0; i < resolution; i++)
    {
        double yvalue = samplingStepSize * i + yCenter - offest;
        for(int j = 0; j < resolution; j++)
        {
            double xvalue = samplingStepSize * j - xCenter - offest;
            // ( (abs(cos(an)*(j-yelements/2-yc-0.5)+sin(an)*(k-xelements/2-xc-0.5))>wdy) |
            // (abs(-sin(an)*(j-yelements/2-yc-0.5)+cos(an)*(k-xelements/2-xc-0.5))>wdx));
            double tmp1 = cosAngle*yvalue + sinAngle*xvalue;
            double tmp2 = -sinAngle*yvalue + cosAngle*xvalue;
            fourierTransmittanceMatrix->set(i, j, (std::abs(tmp1) > width | std::abs(tmp2) > height));
        }
    }
}
