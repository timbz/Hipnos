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

#ifndef PIPELINE_H
#define PIPELINE_H

#include <QMutex>

#include "pipelinecomponent.h"
#include "pipelinedata.h"

/**
 * @brief This class represents an interface to the simulation pipeline.
 *      Because the PipelineComponents are connected this class has only a reference to the first PipelineComponent.
 */
class Pipeline
{
public:
    explicit Pipeline();
    explicit Pipeline(PipelineComponent* c);
    ~Pipeline();

    static void pipelineConnection(PipelineComponent* from, int i, PipelineComponent* to, int j);
    static void pipelineConnection(PipelineComponent* from, PipelineComponent* to);

    PipelineComponent* getFirstComponent();
    void setFirstComponent(PipelineComponent* c);
    PipelineSpectrumData* propagation(PipelineDataType dataType, double z);
    double getLength();

private:
    PipelineComponent* findComponentAt(double z);

    PipelineComponent* first;
    QSet<PipelineConnection*> connections;  
};

#endif // PIPELINE_H
