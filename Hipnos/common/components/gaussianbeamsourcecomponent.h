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

#ifndef SOURCECOMPONENT_H
#define SOURCECOMPONENT_H

#include "common/pipelinecomponent.h"
#include "common/widgets/spinbox.h"
#include "common/spectrum.h"

/**
 * @brief This PipelineComponent simulates a gaussian beam source.
 *
 * It's an source component so it has no input connections.
 * It created a specific instance of a PipelineData and pushes it down the pipeline
 */
class GaussianBeamSourceComponent : public PipelineComponent
{

public:
    explicit GaussianBeamSourceComponent(Spectrum s = 1.0/(1030 * 0.000000001), double w = 0.002, double r = 0, double e = 1, int res = 128);
    ~GaussianBeamSourceComponent();

    QString getType();
    PipelineComponent* clone();
    QIcon getIcon();
    void setProperty(Property p);
    QList<Property> getProperties();
    void gaussPropagation(double z);
    void fourierPropagation(double z);

private:
    double w0;  
    double phaseradius;  
    double e0;  
    int resolution;  
    Spectrum spectrum;  
    EvenIntValidator resolutionValidator;  
};

#endif // SOURCECOMPONENT_H
