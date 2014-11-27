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

#include "appmlmatrix.h"
#include <QDebug>

AppmlMatrix::AppmlMatrix(cl_context c, cl_command_queue q, int row, int col)
{
    Q_ASSERT(row > 1);
    Q_ASSERT(col > 1);

    event = NULL;
    ctx = c;
    queue = q;
    rows = row;
    cols = col;
    hostRaw = new std::complex<double>[rows*cols];

    deviceRaw = clCreateBuffer(ctx, CL_MEM_READ_WRITE, rows*cols * sizeof(std::complex<double>),
                              NULL, &err);
    checkForError("clCreateBuffer");
    deviceDataChanged = true;
    hostDataChanged = false;
}

AppmlMatrix::~AppmlMatrix()
{
    delete[] hostRaw;
    err = clReleaseMemObject(deviceRaw);
    checkForError("clReleaseMemObject");
}

void AppmlMatrix::checkForError(QString functionName)
{
    if(err != CL_SUCCESS)
    {
        QString error = "Error executing " + functionName + " (code " + QString::number(err) + ")";
        qFatal(error.toStdString().c_str());
    }
}

void* AppmlMatrix::data()
{
    if(hostDataChanged)
        updateDeviceData();
    return deviceRaw;
}

Matrix* AppmlMatrix::clone()
{
    if(hostDataChanged)
        updateDeviceData();
    AppmlMatrix* clone = new AppmlMatrix(ctx, queue, rows, cols);
    err = clEnqueueCopyBuffer(queue, deviceRaw, static_cast<cl_mem>(clone->data()), 0, 0,
                        rows*cols * sizeof(std::complex<double>), 0, NULL, &event);
    checkForError("clEnqueueCopyBuffer");
    err = clWaitForEvents(1, &event);
    checkForError("clWaitForEvents[clEnqueueCopyBuffer]");
    err = clReleaseEvent(event);
    checkForError("clReleaseEvent");
    return clone;
}

std::complex<double> AppmlMatrix::get(int row, int col)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    if(deviceDataChanged)
        updateHostData();
    return hostRaw[col * rows + row];
}

void AppmlMatrix::set(int row, int col, std::complex<double> s)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    if(deviceDataChanged)
        updateHostData();
    hostRaw[col * rows + row] = s;
    hostDataChanged = true;
}

void AppmlMatrix::updateDeviceData()
{
    err = clEnqueueWriteBuffer(queue, deviceRaw, CL_TRUE, 0,
                               rows*cols * sizeof(std::complex<double>), hostRaw, 0, NULL, NULL);
    checkForError("clEnqueueWriteBuffer");
    hostDataChanged = false;
}

void AppmlMatrix::updateHostData()
{
    err = clEnqueueReadBuffer(queue, deviceRaw, CL_TRUE, 0, rows*cols * sizeof(std::complex<double>),
                              hostRaw, 0, NULL, NULL);
    checkForError("clEnqueueReadBuffer");
    deviceDataChanged = false;
}

int AppmlMatrix::getCols()
{
    return cols;
}

int AppmlMatrix::getRows()
{
    return rows;
}

void AppmlMatrix::setDeviceDataChanged()
{
    deviceDataChanged = true;
}
