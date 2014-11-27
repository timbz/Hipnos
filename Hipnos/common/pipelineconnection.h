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

#ifndef PIPELINECONNECTION_H
#define PIPELINECONNECTION_H

#include "pipelinedata.h"

#include <QMutex>

class PipelineComponent;

/**
 * @brief This class connects 2 PipelineComponents together and pulls/pushed data between them
 *
 */
class PipelineConnection
{

public:
    PipelineConnection(PipelineComponent* i, PipelineComponent* o, bool forceDisableCash = false);
    virtual ~PipelineConnection();

    PipelineSpectrumData* fetchIntermediatDataFromInput(PipelineDataType dataType, double z);
    PipelineSpectrumData* getData(PipelineDataType dataType);
    void setData(PipelineSpectrumData* data);
    PipelineSpectrumData* getCurrentData();
    PipelineComponent* getInput();
    void setInput(PipelineComponent* i);
    PipelineComponent* getOutput();
    void setOutput(PipelineComponent* o);
    void flush();

protected:
    void fetchData(PipelineDataType dataType);

    PipelineComponent* in;  
    PipelineComponent* out;  
    bool cached;  
    PipelineSpectrumData* currentData;  
    QMutex* lock;  

};

/**
 * @brief A special type of PipelineConnection that connects a PipelineComponent to NULL.
 *          This is used for the output connections of the last PipelineComponent in a Pipeline
 */
class PipelineSink : public PipelineConnection
{

public:
    /**
     * @brief Created a connection to NULL
     *
     * @param i A pointer to the PipelineComponent
     */
    PipelineSink(PipelineComponent* i) :
        PipelineConnection(i, 0)
    {
        cached = false;
    }
};

#endif // PIPELINECONNECTION_H
