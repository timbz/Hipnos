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

#include "cublasmatrix.h"
#include <QDebug>

CuBlasMatrix::CuBlasMatrix(int row, int col)
{
    Q_ASSERT(row > 1);
    Q_ASSERT(col > 1);

    rows = row;
    cols = col;
    hostRaw = new cuDoubleComplex[row*col];
    cudaError_t status = cudaMalloc((void**)&deviceRaw, sizeof(cuDoubleComplex)*rows*cols);
    if(status != cudaSuccess )
    {
        QString error = "Device memory allocation failed (code " + QString::number(status) + ")";
        qFatal(error.toStdString().c_str());
    }
    deviceDataChanged = true;
    hostDataChanged = false;
}

CuBlasMatrix::~CuBlasMatrix()
{
    delete[] hostRaw;
    cudaFree(deviceRaw);
}

void* CuBlasMatrix::data()
{
    if(hostDataChanged)
        updateDeviceData();
    return deviceRaw;
}

Matrix* CuBlasMatrix::clone()
{
    if(hostDataChanged)
        updateDeviceData();
    CuBlasMatrix* clone = new CuBlasMatrix(rows, cols);
    cudaMemcpy(clone->data(), deviceRaw, sizeof(cuDoubleComplex)*rows*cols, cudaMemcpyDeviceToDevice);
    return clone;
}

std::complex<double> CuBlasMatrix::get(int row, int col)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    if(deviceDataChanged)
        updateHostData();
    return std::complex<double>(hostRaw[col * rows + row].x, hostRaw[col * rows + row].y);
}

void CuBlasMatrix::set(int row, int col, std::complex<double> e)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    if(deviceDataChanged)
        updateHostData();
    cuDoubleComplex tmp = make_cuDoubleComplex(e.real(), e.imag());
    hostRaw[col * rows + row] = tmp;
    hostDataChanged = true;
}

void CuBlasMatrix::updateDeviceData()
{
    cublasStatus_t status = cublasSetMatrix(rows, cols, sizeof(cuDoubleComplex), hostRaw, rows, deviceRaw, rows);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Data upload faild");
    }
    hostDataChanged = false;
}

void CuBlasMatrix::updateHostData()
{
    cublasStatus_t status = cublasGetMatrix(rows, cols, sizeof(cuDoubleComplex), deviceRaw, rows, hostRaw, rows);
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        qFatal("Data download faild");
    }
    deviceDataChanged = false;
}

int CuBlasMatrix::getCols()
{
    return cols;
}

int CuBlasMatrix::getRows()
{
    return rows;
}

void CuBlasMatrix::setDeviceDataChanged()
{
    deviceDataChanged = true;
}
