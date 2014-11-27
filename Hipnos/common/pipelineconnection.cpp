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

#include "pipelineconnection.h"
#include "pipelinecomponent.h"
#include "hipnossettings.h"

PipelineConnection::PipelineConnection(PipelineComponent *i, PipelineComponent *o, bool forceDisableCash)
{
    lock = new QMutex(QMutex::Recursive);
    in = i;
    out = o;
    currentData = 0;
    if(forceDisableCash)
        cached = false;
    else
        cached = HipnosSettings::getInstance().usePipelieCash();
}

PipelineConnection::~PipelineConnection()
{
    delete lock;
    if(currentData)
        delete currentData;
}

void PipelineConnection::flush()
{    
    lock->lock();
    if(currentData)
    {
        delete currentData;
        currentData = 0;
    }
    lock->unlock();
    if(out)
    {
        out->flush();
    }
}

void PipelineConnection::setData(PipelineSpectrumData *data)
{
    lock->lock();
    {
        if(currentData)
            delete currentData;
        // if cached we store the data and return a clone on getData()
        // if not cached we store the data and return it on getData()
        currentData = data;
    }
    lock->unlock();
}

PipelineSpectrumData *PipelineConnection::getData(PipelineDataType dataType)
{
    lock->lock();
    PipelineSpectrumData* toReturn;
    {
        Q_ASSERT(in != 0);
        if(currentData == 0 || currentData->getDataType() != dataType)
        {
            fetchData(dataType);
        }
        if(cached)
        {
            toReturn = currentData->clone();
        }
        else // no cache
        {
            toReturn = currentData;
            currentData = 0;
        }
    }
    lock->unlock();
    return toReturn;
}

PipelineSpectrumData* PipelineConnection::fetchIntermediatDataFromInput(PipelineDataType dataType, double z)
{
    lock->lock();
    // backup current data
    PipelineSpectrumData* bkp = currentData;
    currentData = 0;
    in->computePropagation(dataType, z);
    Q_ASSERT_X(currentData != 0, "PipelineConnection::fetchIntermediatDataFromInput", in->getName().toLocal8Bit() + " faild to update output data");
    Q_ASSERT_X(currentData->getDataType() == dataType, "PipelineConnection::fetchIntermediatDataFromInput", "wrong data type");
    PipelineSpectrumData* result = currentData;
    currentData = bkp;
    lock->unlock();
    return result;
}

void PipelineConnection::fetchData(PipelineDataType dataType)
{
    in->computePropagation(dataType, in->getLength());
    Q_ASSERT_X(currentData != 0, "PipelineConnection::fetchData", in->getName().toLocal8Bit() + " faild to update output data");
    Q_ASSERT_X(currentData->getDataType() == dataType, "PipelineConnection::fetchData", "wrong data type");
}

PipelineComponent* PipelineConnection::getInput()
{
    return in;
}

void PipelineConnection::setInput(PipelineComponent *i)
{
    in = i;
}

PipelineComponent* PipelineConnection::getOutput()
{
    return out;
}

void PipelineConnection::setOutput(PipelineComponent *o)
{
    out = o;
}

PipelineSpectrumData *PipelineConnection::getCurrentData()
{
    return currentData;
}
