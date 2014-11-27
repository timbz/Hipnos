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

#include "gslmatrix.h"
#include <string.h>
#include <QtGlobal>

GslMatrix::GslMatrix(int row, int col)
{
    Q_ASSERT(row > 1);
    Q_ASSERT(col > 1);

    rows = row;
    cols = col;
    raw = new std::complex<double>[row*col];
}

GslMatrix::~GslMatrix()
{
    delete[] raw;
}

void* GslMatrix::data()
{
    return raw;
}

std::complex<double> GslMatrix::get(int row, int col)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    return raw[col * rows + row];
}

void GslMatrix::set(int row, int col, std::complex<double> e)
{
    Q_ASSERT(row < rows);
    Q_ASSERT(col < cols);
    raw[col * rows + row] = e;
}

Matrix* GslMatrix::clone()
{
    GslMatrix* clone = new GslMatrix(rows, cols);
    memcpy(clone->data(), raw, sizeof(std::complex<double>)*rows*cols);
    return clone;
}

int GslMatrix::getCols()
{
    return cols;
}

int GslMatrix::getRows()
{
    return rows;
}
