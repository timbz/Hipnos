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

#ifndef CSVSOURCECOMPONENT_H
#define CSVSOURCECOMPONENT_H

#include "common/pipelinecomponent.h"
#include "common/spectrum.h"

/**
 * @brief This PipelineComponent loads ampltude an phase data from two CSV files
 *
 * It's an source component so it has no input connections.
 * It created a specific instance of a PipelineData from the CSV data and pushes it down the pipeline.
 */
class CsvSourceComponent : public PipelineComponent
{

public:
    explicit CsvSourceComponent(QString iDataPath = "", QString pDataPath = "", QString sep = ", ", Spectrum s = 1.0/0.000001, double step = 0.0001);
    ~CsvSourceComponent();

    QString getType();
    PipelineComponent* clone();
    QIcon getIcon();
    void setProperty(Property p);
    QList<Property> getProperties();
    void gaussPropagation(double z);
    void fourierPropagation(double z);

private:
    QString intensityDataPath;  
    QString phaseDataPath;  
    Spectrum spectrum;  
    double stepSize;  
    QString csvSeparator;  

};
#endif // CSVSOURCECOMPONENT_H
