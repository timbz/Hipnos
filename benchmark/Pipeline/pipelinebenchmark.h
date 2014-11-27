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

#ifndef PIPELINEBENCHMARK_H
#define PIPELINEBENCHMARK_H

#include <QtCore/QString>
#include <QtTest/QtTest>
#include "common/math/math.h"

class PipelineBenchmark : public QObject
{
    Q_OBJECT

public:

    PipelineBenchmark();

    ~PipelineBenchmark();

private Q_SLOTS:

    void testFourierPropagation_data();

    void testFourierPropagation();

private:
    QList<MathPlugin*> plugins;  
};

#endif // PIPELINEBENCHMARK_H
