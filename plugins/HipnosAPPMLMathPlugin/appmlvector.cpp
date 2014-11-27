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

#include "appmlvector.h"
#include <QDebug>

AppmlVector::AppmlVector(cl_context c, cl_command_queue q, int s)
{
    Q_ASSERT(s > 1);

    event = NULL;
    ctx = c;
    queue = q;
    size = s;
    hostRaw = new std::complex<double>[size];

    deviceRaw = clCreateBuffer(ctx, CL_MEM_READ_WRITE, s * sizeof(std::complex<double>),
                              NULL, &err);
    checkForError("clCreateBuffer");
    deviceDataChanged = true;
    hostDataChanged = false;
}

AppmlVector::~AppmlVector()
{
    delete[] hostRaw;
    err = clReleaseMemObject(deviceRaw);
    checkForError("clReleaseMemObject");
}

void AppmlVector::checkForError(QString functionName)
{
    if(err != CL_SUCCESS)
    {
        QString error = "Error executing " + functionName + " (code " + QString::number(err) + ")";
        qFatal(error.toStdString().c_str());
    }
}

void* AppmlVector::data()
{
    if(hostDataChanged)
        updateDeviceData();
    return deviceRaw;
}

Vector* AppmlVector::clone()
{
    if(hostDataChanged)
        updateDeviceData();
    AppmlVector* clone = new AppmlVector(ctx, queue, size);
    err = clEnqueueCopyBuffer(queue, deviceRaw, static_cast<cl_mem>(clone->data()), 0, 0,
                        size * sizeof(std::complex<double>), 0, NULL, &event);
    checkForError("clEnqueueCopyBuffer");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueCopyBuffer]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    return clone;
}

std::complex<double> AppmlVector::get(int i)
{
    Q_ASSERT(i < size);
    if(deviceDataChanged)
        updateHostData();
    return hostRaw[i];
}

void AppmlVector::set(int i, std::complex<double> e)
{
    Q_ASSERT(i < size);
    if(deviceDataChanged)
        updateHostData();
    hostRaw[i] = e;
    hostDataChanged = true;
}

void AppmlVector::updateDeviceData()
{
    err = clEnqueueWriteBuffer(queue, deviceRaw, CL_TRUE, 0,
                               size * sizeof(std::complex<double>), hostRaw, 0, NULL, NULL);
    checkForError("clEnqueueWriteBuffer");
    hostDataChanged = false;
}

void AppmlVector::updateHostData()
{
    err = clEnqueueReadBuffer(queue, deviceRaw, CL_TRUE, 0, size * sizeof(std::complex<double>),
                              hostRaw, 0, NULL, NULL);
    checkForError("clEnqueueReadBuffer");
    deviceDataChanged = false;
}

int AppmlVector::getSize()
{
    return size;
}

void AppmlVector::setDeviceDataChanged()
{
    deviceDataChanged = true;
}
