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

#include "acmlvector.h"
#include <string.h>
#include <QtGlobal>

AcmlVector::AcmlVector(int s)
{
    Q_ASSERT(s > 1);

    size = s;
    raw = new std::complex<double>[size];
}

AcmlVector::~AcmlVector()
{
    delete[] raw;
}

void* AcmlVector::data()
{
    return raw;
}

std::complex<double> AcmlVector::get(int i)
{
    Q_ASSERT(i < size);
    return raw[i];
}

void AcmlVector::set(int i, std::complex<double> s)
{
    Q_ASSERT(i < size);
    raw[i] = s;
}

Vector* AcmlVector::clone()
{
    AcmlVector* clone = new AcmlVector(size);
    memcpy(clone->data(), raw, sizeof(std::complex<double>)*size);
    return clone;
}

int AcmlVector::getSize()
{
    return size;
}
