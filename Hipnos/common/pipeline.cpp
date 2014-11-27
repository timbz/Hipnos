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

#include "pipeline.h"
#include "common/components/gaussianbeamsourcecomponent.h"
#include "common/components/propagationcomponent.h"
#include "common/components/aperturecomponent.h"

#include <QMutexLocker>

Pipeline::Pipeline()
{
    first = NULL;
}

Pipeline::Pipeline(PipelineComponent* p)
{
    setFirstComponent(p);
}

Pipeline::~Pipeline()
{
    foreach(PipelineConnection* con, connections)
    {
        delete con;
    }
}

double Pipeline::getLength()
{
    double l = 0;
    PipelineComponent* c = first;
    while(c)
    {
        l  += c->getLength();
        if(c->getNumberOfOutputConnections() > 0)
            c = c->getOutputConnection()->getOutput();
        else
            c = 0;
    }
    return l;
}

PipelineComponent* Pipeline::getFirstComponent()
{
    return first;
}

void Pipeline::setFirstComponent(PipelineComponent *c)
{
    first = c;
    // We store the connections in the pipeline.
    // Now we can delete the connections in the pipeline
    // destrucor without deleting the components.
    // The components might be reused in the designer
    // Note: QSet doesnt store duplicates
    while(c)
    {
        PipelineComponent* next = c->getOutputConnection()->getOutput();
        for(int i = 0; i < c->getNumberOfInputConnections(); i++)
        {
            connections.insert(c->getInputConnection(i));
        }
        for(int i = 0; i < c->getNumberOfOutputConnections(); i++)
        {
            connections.insert(c->getOutputConnection(i));
        }
        c = next;
    }
}

PipelineComponent* Pipeline::findComponentAt(double z)
{
    PipelineComponent* c = first;
    PipelineComponent* result = 0;
    while(c && z >= 0)
    {
        z -= c->getLength();
        result = c;
        c = c->getOutputConnection()->getOutput();
    }
    return result;
}

PipelineSpectrumData* Pipeline::propagation(PipelineDataType dataType, double z)
{
    PipelineComponent* c = first;
    while(c->getOutputConnection()->getOutput()
          && z >= c->getLength())
    {
        z -= c->getLength();
        c = c->getOutputConnection()->getOutput();
    }
    return c->getOutputConnection()->fetchIntermediatDataFromInput(dataType, z);
}

void Pipeline::pipelineConnection(PipelineComponent* from, int i, PipelineComponent *to, int j)
{
    PipelineConnection* con = new PipelineConnection(from, to);
    from->setOutputConnection(con, i);
    to->setInputConnection(con, j);
}

void Pipeline::pipelineConnection(PipelineComponent* from, PipelineComponent *to)
{
    pipelineConnection(from, 0, to, 0);
}
