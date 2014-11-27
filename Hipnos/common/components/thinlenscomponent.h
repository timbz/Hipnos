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

#ifndef THINLENSCOMPONENT_H
#define THINLENSCOMPONENT_H

#include "common/pipelinecomponent.h"

/**
 * @brief This PipelineComponent simulates the transformation of a thin lens on a PipelineData instance
 *
 */
class ThinLensComponent : public PipelineComponent
{

public:
    explicit ThinLensComponent(double f = 25, double x  = 0, double y = 0);
    ~ThinLensComponent();

    QString getType();
    PipelineComponent* clone();
    QIcon getIcon();
    void setProperty(Property p);
    QList<Property> getProperties();
    void gaussPropagation(double z);
    void fourierPropagation(double z);

private:
    double focalLength;  
    double xCenter;  
    double yCenter;  
};

#endif // THINLENSCOMPONENT_H
