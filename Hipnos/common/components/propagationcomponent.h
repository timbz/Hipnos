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

#ifndef DRIVTCOMPONENT_H
#define DRIVTCOMPONENT_H

#include "common/pipelinecomponent.h"
#include <QMutex>

/**
 * @brief This PipelineComponent simulates free space propagation
 *
 */
class PropagationComponent : public PipelineComponent
{
public:

    /**
     * @brief Specifies the propagation method used by the PropagationComponent
     *
     */
    enum PropagationMethod
    {
        PM_NEAR_FIELD,
        PM_FAR_FIELD
    };

    explicit PropagationComponent(double l = 100, PropagationMethod pm = PM_NEAR_FIELD);
    ~PropagationComponent();

    QString getType();
    QIcon getIcon();
    PipelineComponent* clone();
    void setProperty(Property p);
    QList<Property> getProperties();
    void gaussPropagation(double z);
    void fourierPropagation(double z);

private:
    PropagationMethod propagationMethod;  
};

#endif // DRIVTCOMPONENT_H
