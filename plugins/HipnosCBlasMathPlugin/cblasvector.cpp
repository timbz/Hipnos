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

#include "cblasvector.h"
#include <string.h>
#include <fftw3.h>
#include <QtGlobal>

CBlasVector::CBlasVector(int s)
{
    Q_ASSERT(s > 1);

    size = s;
    raw = (std::complex<double>*) fftw_malloc(sizeof(std::complex<double>) * size);
}

CBlasVector::~CBlasVector()
{
    fftw_free(raw);
}

void* CBlasVector::data()
{
    return raw;
}

std::complex<double> CBlasVector::get(int i)
{
    Q_ASSERT(i < size);
    return raw[i];
}

void CBlasVector::set(int i, std::complex<double> s)
{
    Q_ASSERT(i < size);
    raw[i] = s;
}

Vector* CBlasVector::clone()
{
    CBlasVector* clone = new CBlasVector(size);
    memcpy(clone->data(), raw, sizeof(std::complex<double>)*size);
    return clone;
}

int CBlasVector::getSize()
{
    return size;
}
