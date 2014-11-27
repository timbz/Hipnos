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

#ifndef SCRIPTEDCOMPONENT_H
#define SCRIPTEDCOMPONENT_H

#include "common/pipelinecomponent.h"

/**
 * @brief Component used to load scriptet optical tranformation (NOT DONE JET)
 *
 */
class ScriptedComponent : public PipelineComponent
{

public:
    ScriptedComponent();

//    QString getType();
//    PipelineComponent* clone();
//    QList<Property> getProperties();
//    void setProperty(Property p);

//protected:
//    void gaussPropagation(double z);
//    void fourierPropagation(double z);
};

#endif // SCRIPTEDCOMPONENT_H
