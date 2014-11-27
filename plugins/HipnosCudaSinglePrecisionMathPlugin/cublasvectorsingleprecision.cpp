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

#include "cublasvectorsingleprecision.h".h"
#include <QDebug>

CuBlasVectorSinglePrecision::CuBlasVectorSinglePrecision(int s)
{
    Q_ASSERT(s > 1);

    size = s;
    hostRaw = new cuComplex[s];
    cudaError_t status = cudaMalloc((void**)&deviceRaw, sizeof(cuComplex)*size);
    if(status != cudaSuccess )
    {
        QString error = "Device memory allocation failed (code " + QString::number(status) + ")";
        qFatal(error.toStdString().c_str());
    }
    deviceDataChanged = true;
    hostDataChanged = false;
}

CuBlasVectorSinglePrecision::~CuBlasVectorSinglePrecision()
{
    delete[] hostRaw;
    cudaFree(deviceRaw);
}

void* CuBlasVectorSinglePrecision::data()
{
    if(hostDataChanged)
        updateDeviceData();
    return deviceRaw;
}

Vector* CuBlasVectorSinglePrecision::clone()
{
    if(hostDataChanged)
        updateDeviceData();
    CuBlasVectorSinglePrecision* clone = new CuBlasVectorSinglePrecision(size);
    cudaMemcpy(clone->data(), deviceRaw, sizeof(cuComplex)*size, cudaMemcpyDeviceToDevice);
    return clone;
}

std::complex<double> CuBlasVectorSinglePrecision::get(int i)
{
    Q_ASSERT(i < size);
    if(deviceDataChanged)
        updateHostData();
    return std::complex<double>(hostRaw[i].x, hostRaw[i].y);
}

void CuBlasVectorSinglePrecision::set(int i, std::complex<double> e)
{
    Q_ASSERT(i < size);
    if(deviceDataChanged)
        updateHostData();
    cuComplex tmp;
    tmp.x = e.real();
    tmp.y = e.imag();
    hostRaw[i] = tmp;
    hostDataChanged = true;
}

void CuBlasVectorSinglePrecision::updateDeviceData()
{
    //qDebug() << "CuBlasVectorSinglePrecision: syncing device data ...";
    cublasStatus_t status = cublasSetVector(size, sizeof(cuComplex), hostRaw, 1, deviceRaw, 1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Data upload faild");
    }
    hostDataChanged = false;
}

void CuBlasVectorSinglePrecision::updateHostData()
{
    //qDebug() << "CuBlasVectorSinglePrecision: syncing host data ...";
    cublasStatus_t status = cublasGetVector(size, sizeof(cuComplex), deviceRaw, 1, hostRaw, 1);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Data download faild");
    }
    deviceDataChanged = false;
}

int CuBlasVectorSinglePrecision::getSize()
{
    return size;
}

void CuBlasVectorSinglePrecision::setDeviceDataChanged()
{
    deviceDataChanged = true;
}
