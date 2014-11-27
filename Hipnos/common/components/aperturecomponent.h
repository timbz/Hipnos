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

#ifndef APERTURECOMPONENT_H
#define APERTURECOMPONENT_H

#include "common/pipelinecomponent.h"

/**
 * @brief Base class for all aperture components
 *
 */
class AbstractApertureComponent : public PipelineComponent
{

public:
    explicit AbstractApertureComponent(double xw, double yw, double x, double yc, double a);
    virtual ~AbstractApertureComponent();
    virtual void setProperty(Property p);
    virtual QList<Property> getProperties();
    virtual void gaussPropagation(double z);
    virtual void fourierPropagation(double z);
    Matrix* getFourierTransmittanceMatrix();
    virtual void updateFourierTransmittanceMatrix(int resolution, double samplingStepSize) = 0;

protected:
    Matrix* fourierTransmittanceMatrix;  
    double fourierTransmittanceMatrixSamplingStep;  
    double width;  
    double height;  
    double xCenter;  
    double yCenter;  
    double angle;  
};

/**
 * @brief Simulates a circular aperture
 *
 */
class CircularApertureComponent : public AbstractApertureComponent
{

public:
    explicit CircularApertureComponent(double xw = 0.004, double yw = 0.004,
                               double xc = 0, double yc = 0,double a = 0) :
        AbstractApertureComponent(xw, yw, xc, yc, a){}

    QIcon getIcon();
    QString getType();
    PipelineComponent* clone();
    void updateFourierTransmittanceMatrix(int resolution, double samplingStepSize);
};

/**
 * @brief Simulates a circular obstacle
 *
 */
class CircularObstacleComponent : public AbstractApertureComponent
{

public:
    explicit CircularObstacleComponent(double xw = 0.004, double yw = 0.004,
                               double xc = 0, double yc = 0,double a = 0) :
        AbstractApertureComponent(xw, yw, xc, yc, a){}

    QIcon getIcon();
    QString getType();
    PipelineComponent* clone();
    void updateFourierTransmittanceMatrix(int resolution, double samplingStepSize);
};

/**
 * @brief Simulates a rectangular aperture
 *
 */
class RectangularApertureComponent : public AbstractApertureComponent
{

public:
    explicit RectangularApertureComponent(double xw = 0.004, double yw = 0.004,
                               double xc = 0, double yc = 0,double a = 0) :
        AbstractApertureComponent(xw, yw, xc, yc, a){}

    QIcon getIcon();
    QString getType();
    PipelineComponent* clone();
    void updateFourierTransmittanceMatrix(int resolution, double samplingStepSize);
};

/**
 * @brief Simulates a rectangular obstacle
 *
 */
class RectangularObstacleComponent : public AbstractApertureComponent
{

public:
    explicit RectangularObstacleComponent(double xw = 0.004, double yw = 0.004,
                               double xc = 0, double yc = 0,double a = 0) :
        AbstractApertureComponent(xw, yw, xc, yc, a){}

    QIcon getIcon();
    QString getType();
    PipelineComponent* clone();
    void updateFourierTransmittanceMatrix(int resolution, double samplingStepSize);
};

#endif // APERTURECOMPONENT_H
